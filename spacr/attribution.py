"""Attribution — what a trained classifier attends to, and whether that means anything.

spaCR already drew Grad-CAM and saliency maps (``spacr.utils.GradCAMGenerator``,
``spacr.utils.SaliencyMapGenerator``, ``spacr.utils.IntegratedGradients``). This
module is the library those two were the first entries in: the CAM family
(Grad-CAM, Grad-CAM++, Score-CAM, XGrad-CAM, Layer-CAM, Eigen-CAM), the gradient
family (saliency, integrated gradients, guided backprop, input×gradient,
DeepLIFT, and SmoothGrad wrapped around any of them), the perturbation family
(occlusion, feature ablation) and attention rollout for transformer backbones
that have no convolution for a CAM to hook.

Almost none of it is written here. The CAM variants come from **torchcam**, the
gradient and perturbation methods from **captum**, both already hard
dependencies. What is written here is the part no library can supply: the
adapters that make every method agree on one output shape, the handling of
spaCR's two classifier head shapes, and — the reason this module exists — the
analyses in the second half.

**An attribution map is not an explanation.** It is a number per pixel produced
by a procedure. It does not show what the model "looked at", it does not
establish that the highlighted pixels caused the prediction, and it will render
a confident, beautiful, plausible picture for a model with random weights. Four
checks stand between a map and wishful thinking, and they are the point of this
module:

* :func:`deletion_curve` / :func:`insertion_curve` — remove (or add) the pixels
  the map ranks highest and watch the score. A map whose deletion curve is flat
  is not describing what the model uses, however good it looks.
* :func:`pointing_game` — does the map's peak land inside the object at all?
  spaCR has the object masks already (``merged/*.npy``), so this costs nothing.
* :func:`randomization_sanity_check` — Adebayo et al. 2018. Randomise the
  model's weights layer by layer and attribute again. Several popular methods
  return a nearly identical map for a randomised model, which means they are
  edge detectors that happen to be plotted over a classifier. This is the single
  most informative check here and the one most often skipped.
* :func:`method_agreement` — rank correlation between methods on the same image.
  Agreement is weak evidence. Disagreement is strong evidence that no single map
  should be trusted.

Every criterion measures a *different* property and they routinely disagree.
None of them is ground truth, because for attribution there is none.

**Two head shapes, one contract.** A spaCR classifier's head emits either one
logit (binary; class 1 when the logit is positive) or ``C`` logits. Code that
assumes one shape is wrong for the other, and silently so: attributing the raw
logit of a single-logit head always explains *class 1*, so for an image the model
called class 0 the map you get is the map for the class it rejected — the
negation of what you asked for. Every method here goes through
:class:`ClassScoreModel`, which presents a single logit ``z`` as the two-column
view ``[-z, +z]``. Both classes then have a real gradient, ``target`` means the
same thing for both head shapes, and no caller has to know which head it has.

:author: spaCR
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Attribution",
    "ATTRIBUTION_METHODS",
    "MethodSpec",
    "AttributionError",
    "NoSpatialLayerError",
    "UnknownMethodError",
    "ClassScoreModel",
    "attribute",
    "smoothgrad",
    "compare_methods",
    "attention_rollout",
    "list_methods",
    "methods_by_family",
    "conv_layer_names",
    "resolve_layer",
    "recommended_layer",
    "class_scores",
    "Curve",
    "deletion_curve",
    "insertion_curve",
    "faithfulness",
    "pointing_game",
    "pointing_game_rate",
    "SanityCheck",
    "randomization_sanity_check",
    "Agreement",
    "method_agreement",
    "AttributionMapGenerator",
    "NOT_AN_EXPLANATION",
    "CRITERION_CAVEATS",
]


# ---------------------------------------------------------------------------
# Messages surfaced verbatim by the GUI, the CLI and the search
# ---------------------------------------------------------------------------

#: Attached to every reported attribution result. Never suppressed.
NOT_AN_EXPLANATION = (
    "An attribution map is not an explanation of causation. It is a number per "
    "pixel produced by a procedure, and every method here will render a "
    "confident-looking map for a model with random weights. Read the deletion "
    "and insertion curves, the pointing-game hit rate and the randomisation "
    "sanity check before believing any of it."
)

#: What each search criterion rewards — and what it cannot see.
CRITERION_CAVEATS: Dict[str, str] = {
    "deletion_auc": (
        "removes the highest-ranked pixels first and averages the class score "
        "along the way, so a LOWER value is better: the score collapsed as soon "
        "as the map's top pixels went. It is confounded by the removal baseline "
        "— blanking a region creates an edge the model has never seen, and part "
        "of the score drop is that artefact rather than the information removed."
    ),
    "insertion_auc": (
        "starts from a blanked image and adds the highest-ranked pixels first, "
        "so a HIGHER value is better. It rewards maps that concentrate on a "
        "small sufficient region, and it systematically favours smooth, blobby "
        "maps over sharp per-pixel ones, which is why it can rank the methods in "
        "the opposite order to deletion."
    ),
    "pointing_game": (
        "asks only whether the map's single brightest pixel falls inside the "
        "object mask. It is cheap and spaCR-specific, it says nothing about the "
        "rest of the map, and it is trivially satisfied by any method biased "
        "towards bright or textured regions when the object is the bright thing "
        "in the frame."
    ),
    "sanity_gap": (
        "one minus the rank correlation between the map from the trained model "
        "and the map from the same model with randomised weights, so a HIGHER "
        "value is better. A method scoring near zero produced the same picture "
        "for a random model and is an edge detector, not an explanation. It "
        "measures dependence on the weights, not correctness."
    ),
}


class AttributionError(RuntimeError):
    """Base class for the failures this module reports instead of guessing."""


class UnknownMethodError(AttributionError):
    """Raised for a method name that is not registered."""


class NoSpatialLayerError(AttributionError):
    """Raised when a CAM is asked of a model that has no spatial layer to hook.

    A CAM is a weighted sum of one convolutional layer's feature maps. A pure
    transformer has no such layer, and hooking its patch embedding produces a
    picture that is not a CAM of anything. Rather than return that picture, the
    CAM adapters raise this and name the model so the caller can switch to
    :func:`attention_rollout` or to a gradient / perturbation method, which work
    on any architecture.
    """


# ---------------------------------------------------------------------------
# Head-shape handling: one logit or C logits, one contract
# ---------------------------------------------------------------------------

class ClassScoreModel(nn.Module):
    """Present any spaCR classifier head as ``C >= 2`` per-class scores.

    A head emitting one logit ``z`` is presented as ``[-z, +z]``: column 1 is
    the evidence for class 1, column 0 the evidence for class 0, and
    ``argmax`` reproduces the ``z > 0`` rule spaCR's binary models use. The
    obvious alternative, ``[0, z]``, is exactly equivalent under softmax but has
    zero gradient for class 0, so every attribution for class 0 would be an
    all-zero map — a silent, plausible-looking wrong answer.

    A head emitting ``C > 1`` logits is passed through untouched.

    :param model: the classifier to wrap.
    :ivar n_out: the wrapped model's raw output width (1 or C).
    :ivar n_classes: the number of classes the wrapper exposes (2 or C).
    :ivar single_logit: True when the wrapped head emits one logit.
    """

    def __init__(self, model: nn.Module, n_out: Optional[int] = None):
        """Wrap ``model``, recording whether its head is single-logit."""
        super().__init__()
        self.model = model
        self.n_out = int(n_out) if n_out is not None else None
        self.single_logit = None if n_out is None else (int(n_out) == 1)

    def _note_width(self, raw: torch.Tensor) -> torch.Tensor:
        """Record the head width the first time a real forward pass reveals it."""
        if raw.ndim == 1:
            raw = raw.unsqueeze(-1)
        if self.n_out is None:
            self.n_out = int(raw.shape[-1])
            self.single_logit = self.n_out == 1
        return raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, n_classes)`` scores for ``x``."""
        raw = self._note_width(self.model(x))
        if raw.shape[-1] == 1:
            return torch.cat([-raw, raw], dim=-1)
        return raw

    @property
    def n_classes(self) -> int:
        """How many classes the wrapper exposes."""
        if self.n_out is None:
            raise AttributionError(
                "the head width is not known yet — run one forward pass "
                "through the wrapper first")
        return 2 if self.n_out == 1 else int(self.n_out)


def _to_batch(image: Any) -> torch.Tensor:
    """Coerce an image into a float ``(1, C, H, W)`` tensor.

    :param image: tensor or array shaped ``(H, W)``, ``(C, H, W)`` or
        ``(1, C, H, W)``.
    :returns: the batched tensor, detached and float.
    :raises AttributionError: for a batch of more than one image or a rank the
        adapters cannot interpret.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(np.asarray(image))
    x = image.detach().float()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim != 4:
        raise AttributionError(
            f"image must be (H, W), (C, H, W) or (1, C, H, W); got shape "
            f"{tuple(image.shape)}.")
    if x.shape[0] != 1:
        raise AttributionError(
            f"attribute() explains one image at a time; got a batch of "
            f"{x.shape[0]}. Loop over the batch, or use compare_methods() per "
            f"image — the analyses downstream are per-image too.")
    return x


def class_scores(model: nn.Module, x: torch.Tensor,
                 *, probability: bool = True) -> torch.Tensor:
    """Per-class scores for ``x``, for either head shape.

    :param model: the classifier (raw or already wrapped).
    :param x: input batch ``(B, C, H, W)``.
    :param probability: return softmax probabilities rather than raw scores.
        Probabilities are what the deletion / insertion curves track, because a
        bounded quantity makes their areas comparable across images.
    :returns: ``(B, n_classes)`` tensor.
    """
    wrapped = model if isinstance(model, ClassScoreModel) else ClassScoreModel(model)
    with torch.no_grad():
        scores = wrapped(x)
    if probability:
        return torch.softmax(scores, dim=-1)
    return scores


def _predicted_class(wrapped: ClassScoreModel, x: torch.Tensor) -> int:
    """The class the model predicts for ``x``, for either head shape."""
    with torch.no_grad():
        return int(wrapped(x).argmax(dim=-1)[0])


def _resolve_target(wrapped: ClassScoreModel, x: torch.Tensor,
                    target: Optional[int]) -> int:
    """Validate ``target`` against the head, defaulting to the prediction."""
    predicted = _predicted_class(wrapped, x)
    if target is None:
        return predicted
    target = int(target)
    n = wrapped.n_classes
    if not 0 <= target < n:
        raise AttributionError(
            f"target={target} is not a class of this model. Its head emits "
            f"{wrapped.n_out} logit(s), which is "
            f"{'a binary head with classes 0 and 1' if wrapped.n_out == 1 else f'{n} classes 0..{n - 1}'}."
        )
    return target


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def conv_layer_names(model: nn.Module) -> List[str]:
    """Every ``Conv2d`` layer name in ``model``, in definition order.

    :param model: the model to scan.
    :returns: dotted layer names; empty for a model with no convolutions.
    """
    return [name for name, mod in model.named_modules()
            if isinstance(mod, nn.Conv2d)]


def recommended_layer(model: nn.Module) -> Optional[str]:
    """The last convolutional layer — the usual CAM target — or None.

    Mirrors :func:`spacr.utils.recommend_target_layers`, but returns None for a
    model with no convolutions instead of raising, so the CAM adapters can raise
    :class:`NoSpatialLayerError` with the architecture named.
    """
    names = conv_layer_names(model)
    return names[-1] if names else None


def resolve_layer(model: nn.Module, name: str) -> nn.Module:
    """Resolve a dotted layer name against ``model``.

    :param model: the model to look in.
    :param name: dotted module path, e.g. ``'features.2'``.
    :returns: the submodule.
    :raises AttributionError: naming the closest available layers. A wrong
        target layer is the most common way a CAM run dies, and a bare
        ``AttributeError: 'Sequential' object has no attribute 'conv_b'`` does
        not tell the user what to type instead.
    """
    modules = dict(model.named_modules())
    if name in modules:
        return modules[name]
    convs = conv_layer_names(model)
    candidates = convs or [n for n in modules if n]
    shown = candidates[-25:] if len(candidates) > 25 else candidates
    more = (f" (and {len(candidates) - len(shown)} earlier ones)"
            if len(candidates) > len(shown) else "")
    kind = "convolutional layers" if convs else "layers"
    raise AttributionError(
        f"target layer {name!r} does not exist in this model. Available "
        f"{kind}{more}: {shown}. The last one, {candidates[-1]!r}, is the "
        f"usual CAM target."
        if candidates else
        f"target layer {name!r} does not exist in this model, and the model "
        f"has no named submodules to target."
    )


def _spatial_target_layer(model: nn.Module, layer: Optional[str],
                          model_type: Optional[str]) -> Tuple[str, nn.Module]:
    """Pick and validate the layer a CAM will hook.

    :param model: the raw (unwrapped) model.
    :param layer: dotted layer name, or None to use the last convolution.
    :param model_type: the architecture name, used in the error message.
    :returns: ``(name, module)``.
    :raises NoSpatialLayerError: when the model has no convolution to hook.
    :raises AttributionError: when a named layer does not exist.
    """
    if layer:
        return str(layer), resolve_layer(model, str(layer))
    name = recommended_layer(model)
    if name is None:
        raise NoSpatialLayerError(
            f"model_type={model_type or type(model).__name__!r} has no Conv2d "
            f"layer, so there is no feature map for a CAM to weight and no "
            f"honest CAM to compute. Use method='attention_rollout' if this is "
            f"a transformer with torch MultiheadAttention blocks, or any of the "
            f"gradient / perturbation methods (saliency, integrated_gradients, "
            f"occlusion, feature_ablation), which need no spatial layer.")
    return name, resolve_layer(model, name)


def _is_attention_module(module: nn.Module) -> bool:
    """Whether a module is an attention block, by type or by class name.

    ``nn.MultiheadAttention`` catches spaCR's and torch's own blocks; the name
    test catches torchvision's and timm's, which subclass ``nn.Module``
    directly (``WindowAttention``, ``RelativePositionalMultiHeadAttention``, ...).
    """
    if isinstance(module, nn.MultiheadAttention):
        return True
    return type(module).__name__.endswith("Attention")


def _check_spatial_activation(module: nn.Module, wrapped: ClassScoreModel,
                              x: torch.Tensor, layer_name: str,
                              model_type: Optional[str],
                              allow_pre_attention: bool = False) -> None:
    """Refuse to CAM a layer that cannot carry a CAM's meaning.

    Two ways that happens, both of which otherwise render a plausible picture:

    * the layer's output is not a ``(B, C, H, W)`` feature map. A transformer
      block emits ``(B, tokens, channels)``; reduced over the channel axis and
      reshaped it makes an image, and that image is not a CAM of anything.
    * the layer is a **patch embedding** — it runs before every attention block
      in the model, so no class-discriminative information has reached it yet.
      This is the pure-ViT trap: ``recommend_target_layers`` happily returns the
      patch-embed ``Conv2d`` because it *is* a convolution, and Grad-CAM over it
      is a picture of local image statistics. Hybrids like MaxViT are unaffected
      — their MBConv layers run after attention blocks, which is why spaCR's
      default MaxViT target layer keeps working.

    :param allow_pre_attention: opt out of the second check when the caller
        really does want the patch embedding.
    """
    order: List[str] = []
    captured: List[torch.Tensor] = []

    def _target_hook(_m, _inp, out):
        """Capture the target layer's output and its position in the pass."""
        captured.append(out if isinstance(out, torch.Tensor) else out[0])
        order.append("target")

    def _attn_hook(_m, _inp, _out):
        """Record that an attention block ran."""
        order.append("attention")

    handles = [module.register_forward_hook(_target_hook)]
    handles += [m.register_forward_hook(_attn_hook)
                for m in wrapped.model.modules()
                if m is not module and _is_attention_module(m)]
    try:
        with torch.no_grad():
            wrapped(x)
    finally:
        for handle in handles:
            handle.remove()

    if not captured:
        raise AttributionError(
            f"target layer {layer_name!r} never ran during the forward pass, so "
            f"it cannot be the CAM target. Check that the layer is actually on "
            f"the path this model takes.")
    act = captured[0]
    if act.ndim != 4:
        raise NoSpatialLayerError(
            f"target layer {layer_name!r} of "
            f"model_type={model_type or type(wrapped.model).__name__!r} emits a "
            f"{act.ndim}-D tensor {tuple(act.shape)}, not a (B, C, H, W) feature "
            f"map. A CAM over it would be a reshaped token vector drawn as an "
            f"image, which means nothing. Use method='attention_rollout' for a "
            f"transformer, or a gradient / perturbation method.")

    if allow_pre_attention or "attention" not in order:
        return
    first_target = order.index("target")
    first_attention = order.index("attention")
    if first_target < first_attention:
        raise NoSpatialLayerError(
            f"target layer {layer_name!r} of "
            f"model_type={model_type or type(wrapped.model).__name__!r} runs "
            f"before every one of the model's {order.count('attention')} "
            f"attention blocks, so it is the patch embedding: no "
            f"class-discriminative information has reached it and a CAM over it "
            f"is a picture of local image statistics, not of what the model "
            f"attends to. Use method='attention_rollout' for this architecture, "
            f"a gradient or perturbation method, or name a later layer "
            f"explicitly. Pass allow_pre_attention=True to override.")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class Attribution:
    """One attribution map plus everything needed to judge it.

    :ivar method: registered method name.
    :ivar map: 2-D ``(H, W)`` float32 array at the input's spatial resolution,
        guaranteed finite. Larger means "ranked higher by this method"; the
        scale is arbitrary and differs between methods, which is why every
        analysis here uses ranks rather than values.
    :ivar raw: the method's signed, per-channel output where it has one
        (gradient and perturbation families), else None. The CAM family has no
        per-channel form.
    :ivar target: the class index the map explains.
    :ivar n_classes: how many classes the head exposes (2 for a single logit).
    :ivar single_logit: whether the underlying head emits one logit.
    :ivar predicted: the class the model actually predicted for this input.
    :ivar layer: the layer a CAM hooked, else None.
    :ivar family: ``'cam'``, ``'gradient'``, ``'perturbation'`` or
        ``'attention'``.
    :ivar backend: which library produced it — ``'torchcam'``, ``'captum'`` or
        ``'spacr'``.
    :ivar params: the keyword arguments that produced this map.
    :ivar notes: caveats the caller must surface.
    """

    method: str
    map: np.ndarray
    target: int
    n_classes: int
    single_logit: bool
    predicted: int
    raw: Optional[np.ndarray] = None
    layer: Optional[str] = None
    family: str = ""
    backend: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial shape of the map."""
        return tuple(self.map.shape)  # type: ignore[return-value]

    def normalized(self) -> np.ndarray:
        """The map rescaled to ``[0, 1]``; all-zero when the map is flat.

        A flat map is exactly what a fully suppressed CAM produces, and the
        unguarded min-max rescale of one is ``0/0``.
        """
        m = np.asarray(self.map, dtype=np.float64)
        lo = float(m.min())
        rng = float(m.max()) - lo
        if not np.isfinite(rng) or rng <= 0:
            return np.zeros_like(m, dtype=np.float32)
        return ((m - lo) / rng).astype(np.float32)

    def peak(self) -> Tuple[int, int]:
        """``(row, col)`` of the single highest-ranked pixel."""
        idx = int(np.argmax(np.asarray(self.map)))
        return divmod(idx, int(self.map.shape[1]))

    def is_flat(self) -> bool:
        """True when the map has no variation and therefore ranks nothing."""
        m = np.asarray(self.map, dtype=np.float64)
        return not np.isfinite(m).all() or float(m.max() - m.min()) <= 0.0


def _finite_2d(values: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    """Reduce an attribution tensor to a finite ``(H, W)`` float32 array.

    Channels are collapsed by summing the absolute per-channel attribution —
    the convention spaCR's ``saliency_image`` already used — and the result is
    resampled to the input's spatial size when the method worked at a coarser
    resolution.
    """
    t = values.detach().float()
    while t.ndim > 3:
        t = t[0]
    if t.ndim == 3:
        t = t.abs().sum(dim=0)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if tuple(t.shape) != tuple(size):
        t = F.interpolate(t[None, None], size=size, mode="bilinear",
                          align_corners=False)[0, 0]
    return t.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Adapters — torchcam
# ---------------------------------------------------------------------------

_TORCHCAM_CLASSES = {
    "gradcam": "GradCAM",
    "gradcam_pp": "GradCAMpp",
    "scorecam": "ScoreCAM",
    "xgradcam": "XGradCAM",
    "layercam": "LayerCAM",
}


def _torchcam_cam(spec: "MethodSpec", wrapped: ClassScoreModel,
                  x: torch.Tensor, target: int, layer: Optional[str],
                  model_type: Optional[str], **kw) -> Tuple[np.ndarray, None, str, List[str]]:
    """Run one torchcam CAM extractor against the wrapped model.

    torchcam owns the maths for all five variants; this adapter only picks the
    target layer, keeps the extractor's hooks off the model afterwards, and
    resamples the CAM to the input resolution. The extractor is built on the
    *wrapped* model so ``class_idx`` means the same class for a single-logit
    head as for a C-logit one.
    """
    import torchcam.methods as tcm

    layer_name, module = _spatial_target_layer(wrapped.model, layer, model_type)
    _check_spatial_activation(module, wrapped, x, layer_name, model_type,
                              bool(kw.get("allow_pre_attention", False)))
    cls = getattr(tcm, _TORCHCAM_CLASSES[spec.name])
    extractor_kw: Dict[str, Any] = {"input_shape": tuple(x.shape[1:])}
    if spec.name == "scorecam":
        extractor_kw["batch_size"] = int(kw.get("batch_size", 8))
    with cls(wrapped, target_layer=module, **extractor_kw) as extractor:
        scores = wrapped(x)
        cams = extractor(int(target), scores)
    cam = cams[0]
    return (_finite_2d(cam, (int(x.shape[-2]), int(x.shape[-1]))), None,
            layer_name, [])


def _eigen_cam(spec: "MethodSpec", wrapped: ClassScoreModel, x: torch.Tensor,
               target: int, layer: Optional[str], model_type: Optional[str],
               **kw) -> Tuple[np.ndarray, None, str, List[str]]:
    """Eigen-CAM: the first principal component of the target layer's activations.

    torchcam 0.4 ships every CAM variant except this one, so the eight lines of
    SVD are here. Eigen-CAM uses no gradients and no class index at all, which
    is worth knowing before reading one: **the map is identical for every
    class**, so it cannot be evidence that the model separated the classes. It
    is included because that same property makes it a useful control — a
    class-conditional method whose map matches Eigen-CAM is not being
    class-conditional.
    """
    layer_name, module = _spatial_target_layer(wrapped.model, layer, model_type)
    _check_spatial_activation(module, wrapped, x, layer_name, model_type,
                              bool(kw.get("allow_pre_attention", False)))
    captured: List[torch.Tensor] = []

    def _hook(_m, _inp, out):
        """Capture the target layer's feature maps."""
        captured.append(out if isinstance(out, torch.Tensor) else out[0])

    handle = module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            wrapped(x)
    finally:
        handle.remove()
    act = captured[0][0]                       # (C, H, W)
    c, h, w = act.shape
    flat = act.reshape(c, h * w).T             # (H*W, C)
    flat = flat - flat.mean(dim=0, keepdim=True)
    # full_matrices=False keeps this cheap for wide feature maps; a rank-0
    # (all-constant) activation makes SVD degenerate, so fall back to the mean.
    try:
        _u, _s, vh = torch.linalg.svd(flat.double(), full_matrices=False)
        proj = (flat.double() @ vh[0]).reshape(h, w)
    except Exception:
        proj = act.mean(dim=0).double()
    if float(proj.sum()) < 0:                  # sign of a singular vector is free
        proj = -proj
    proj = proj - proj.min()
    return (_finite_2d(proj.float(), (int(x.shape[-2]), int(x.shape[-1]))),
            None, layer_name,
            ["Eigen-CAM ignores the class and the gradients entirely: the same "
             "map is returned for every target, so it cannot show that the "
             "model distinguished the classes."])


# ---------------------------------------------------------------------------
# Adapters — captum
# ---------------------------------------------------------------------------

def _captum_baseline(kind: Any, x: torch.Tensor) -> torch.Tensor:
    """Build the reference input a baseline-dependent method integrates from.

    :param kind: ``'zero'``, ``'mean'``, ``'blur'``, ``'uniform'``, a number, or
        a tensor broadcastable to ``x``.
    :returns: the baseline tensor.
    :raises AttributionError: for an unknown name.
    """
    if isinstance(kind, torch.Tensor):
        return kind.to(x.dtype).expand_as(x).clone()
    if isinstance(kind, (int, float)) and not isinstance(kind, bool):
        return torch.full_like(x, float(kind))
    name = str(kind or "zero").lower()
    if name in ("zero", "zeros", "black", "none"):
        return torch.zeros_like(x)
    if name in ("mean", "channel_mean"):
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x).clone()
    if name in ("blur", "blurred"):
        return _blur(x)
    if name in ("uniform", "random", "noise"):
        return torch.rand_like(x) * (x.max() - x.min()) + x.min()
    raise AttributionError(
        f"unknown baseline {kind!r}; use 'zero', 'mean', 'blur', 'uniform', a "
        f"number, or a tensor. The baseline is not cosmetic: integrated "
        f"gradients attributes the difference between the input and this "
        f"reference, so a different baseline is a different question.")


def _blur(x: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """Gaussian-blur a batch with a separable depthwise convolution.

    The radius is clamped to the image: reflection padding wider than the
    dimension it pads raises, and spaCR's object crops are routinely smaller
    than the 3-sigma radius a 224-pixel default assumes.
    """
    limit = max(1, min(int(x.shape[-2]), int(x.shape[-1])) - 1)
    radius = max(1, min(int(round(3.0 * float(sigma))), limit))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    kernel = kernel / kernel.sum()
    c = int(x.shape[1])
    out = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"),
                   kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="reflect"),
                   kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return out


def _captum_attribute(spec: "MethodSpec", wrapped: ClassScoreModel,
                      x: torch.Tensor, target: int, layer: Optional[str],
                      model_type: Optional[str], **kw
                      ) -> Tuple[np.ndarray, np.ndarray, None, List[str]]:
    """Run one captum attributor against the wrapped model.

    captum owns the maths; this adapter supplies the arguments each method
    needs, converts spaCR's parameter names, and turns captum's signed
    per-channel output into the module's ``(H, W)`` contract without losing the
    signed form.
    """
    import captum.attr as ca

    notes: List[str] = []
    inp = x.clone().requires_grad_(True)
    name = spec.name
    kwargs: Dict[str, Any] = {"target": int(target)}

    if name == "saliency":
        attributor: Any = ca.Saliency(wrapped)
        kwargs["abs"] = bool(kw.get("abs", True))
    elif name == "integrated_gradients":
        attributor = ca.IntegratedGradients(wrapped)
        kwargs["baselines"] = _captum_baseline(kw.get("baseline", "zero"), x)
        kwargs["n_steps"] = int(kw.get("n_steps", kw.get("ig_steps", 50)))
        if kwargs["n_steps"] < 2:
            raise AttributionError(
                f"integrated gradients needs at least 2 steps to integrate "
                f"anything, got n_steps={kwargs['n_steps']}.")
    elif name == "guided_backprop":
        attributor = ca.GuidedBackprop(wrapped)
        notes.append(
            "Guided backprop is the method most often reported as failing the "
            "randomisation sanity check: its ReLU clamping recovers image edges "
            "almost independently of the weights. Run "
            "randomization_sanity_check() before reading anything into it.")
    elif name == "input_x_gradient":
        attributor = ca.InputXGradient(wrapped)
    elif name == "deeplift":
        attributor = ca.DeepLift(wrapped)
        kwargs["baselines"] = _captum_baseline(kw.get("baseline", "zero"), x)
    elif name == "occlusion":
        attributor = ca.Occlusion(wrapped)
        window = int(kw.get("window", kw.get("occlusion_window", 8)))
        stride = int(kw.get("stride", kw.get("occlusion_stride", max(1, window // 2))))
        c, h, w = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        window = max(1, min(window, h, w))
        stride = max(1, min(stride, window))
        kwargs["sliding_window_shapes"] = (c, window, window)
        kwargs["strides"] = (c, stride, stride)
        kwargs["baselines"] = _captum_baseline(kw.get("baseline", "zero"), x)
        kwargs["show_progress"] = False
    elif name == "feature_ablation":
        attributor = ca.FeatureAblation(wrapped)
        block = int(kw.get("block", kw.get("occlusion_window", 8)))
        kwargs["feature_mask"] = _block_mask(x, block)
        kwargs["baselines"] = _captum_baseline(kw.get("baseline", "zero"), x)
        kwargs["show_progress"] = False
    else:
        raise UnknownMethodError(
            f"{name!r} is registered as a captum method but this adapter has "
            f"no branch for it; the registry and the adapter disagree.")

    # captum warns rather than raises for the conditions that silently corrupt
    # a result — most importantly a model that reuses one ReLU instance across
    # layers, which torchvision's ResNets do and which makes DeepLIFT's rescale
    # rule attribute through the wrong activation. A warning printed once to
    # stderr during a batch job is a warning nobody reads, so it is captured
    # and carried on the result instead.
    import warnings as _warnings
    try:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            if spec.smoothed:
                tunnel = ca.NoiseTunnel(attributor)
                kwargs["nt_type"] = "smoothgrad"
                kwargs["nt_samples"] = int(kw.get("n_samples", 25))
                kwargs["stdevs"] = float(
                    _sigma_to_stdev(kw.get("sigma", 0.15), x))
                result = tunnel.attribute(inp, **kwargs)
            else:
                result = attributor.attribute(inp, **kwargs)
    except RuntimeError as exc:
        # torchvision's ResNets — spaCR's most common backbone after MaxViT —
        # build one `nn.ReLU(inplace=True)` and call it at several points.
        # DeepLIFT's rescale rule needs one hook per activation *use*, so it
        # dies here with a message about module attributes that does not tell a
        # user what to do next.
        if "more than once" in str(exc) or "required for DeepLift" in str(exc):
            raise AttributionError(
                f"{name} cannot run on this model: it reuses one activation "
                f"module (typically a single nn.ReLU(inplace=True)) at several "
                f"points in the forward pass, and DeepLIFT needs a distinct "
                f"module per activation to attach its rescale rule to. "
                f"torchvision's ResNets are built this way. Use "
                f"'integrated_gradients' (the same axiomatic family, no such "
                f"requirement), 'input_x_gradient', or a perturbation method. "
                f"captum reported: {exc}") from exc
        raise
    for warning in caught:
        notes.append(f"captum warning: {warning.message}")

    raw = result.detach()[0].cpu().numpy().astype(np.float32)
    return (_finite_2d(result, (int(x.shape[-2]), int(x.shape[-1]))), raw,
            None, notes)


def _block_mask(x: torch.Tensor, block: int) -> torch.Tensor:
    """Group pixels into ``block × block`` tiles for feature ablation.

    Ablating one pixel at a time on a 224² image is 50 176 forward passes and
    tells you almost nothing, because a single pixel changes no convolutional
    response enough to move the score. Tiles are the usable form.
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    block = max(1, min(int(block), h, w))
    rows = torch.arange(h) // block
    cols = torch.arange(w) // block
    n_cols = int(cols.max()) + 1
    ids = (rows[:, None] * n_cols + cols[None, :]).long()
    return ids[None, None].expand(1, int(x.shape[1]), h, w).contiguous()


def _sigma_to_stdev(sigma: float, x: torch.Tensor) -> float:
    """Convert a SmoothGrad noise fraction into an absolute standard deviation.

    ``sigma`` is a fraction of the input's dynamic range, matching the
    ``stdev_spread`` convention of the SmoothGrad paper and of
    :class:`spacr.deep_spacr.SmoothGrad`. An absolute standard deviation would
    mean something different for a ``[0, 1]`` image and a z-scored one.
    """
    span = float(x.max() - x.min())
    if span <= 0:
        span = 1.0
    return max(float(sigma) * span, 1e-12)


# ---------------------------------------------------------------------------
# Adapter — attention rollout
# ---------------------------------------------------------------------------

def attention_rollout(model: nn.Module, image: Any, *,
                      target: Optional[int] = None,
                      head_fusion: str = "mean",
                      discard_ratio: float = 0.0,
                      model_type: Optional[str] = None) -> Attribution:
    """Attention rollout (Abnar & Zuidema 2020) for transformer backbones.

    A CAM needs a convolutional feature map. A pure transformer has none, so
    this is the substitute: the per-layer attention matrices are averaged over
    heads, mixed with the identity to account for the residual stream, row-
    normalised and multiplied together, giving how much each input token
    contributes to the class token.

    It reads spaCR's :class:`torch.nn.MultiheadAttention` blocks, which return
    their attention weights from ``forward``. Backbones whose attention is a
    fused kernel (timm's ViT via ``scaled_dot_product_attention``) expose no
    weights, and this raises rather than inventing a map.

    **Rollout is not class-conditional.** The result is the same for every
    ``target``: it describes where information flowed, not what the model
    concluded. It cannot show the model separated your classes; a gradient or
    perturbation method can.

    :param model: the transformer classifier.
    :param image: one image, ``(C, H, W)`` or ``(1, C, H, W)``.
    :param target: recorded on the result; does not change the map.
    :param head_fusion: ``'mean'``, ``'max'`` or ``'min'`` over attention heads.
    :param discard_ratio: fraction of the lowest attention weights zeroed per
        layer before rollout, which sharpens the map and is pure cosmetics.
    :param model_type: architecture name for the error messages.
    :returns: the :class:`Attribution`.
    :raises NoSpatialLayerError: when the model exposes no attention weights.
    """
    x = _to_batch(image)
    wrapped = ClassScoreModel(model)
    target = _resolve_target(wrapped, x, target)
    predicted = _predicted_class(wrapped, x)

    blocks = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
    if not blocks:
        convs = conv_layer_names(model)
        raise NoSpatialLayerError(
            f"model_type={model_type or type(model).__name__!r} exposes no "
            f"torch.nn.MultiheadAttention block, so there are no attention "
            f"weights to roll out. "
            + (f"It does have convolutional layers ({convs[-1]!r} last), so a "
               f"CAM method applies instead."
               if convs else
               "It has no Conv2d layer either, so no CAM applies — use a "
               "gradient or perturbation method (saliency, "
               "integrated_gradients, occlusion), which need neither.")
            + " Backbones whose attention is a fused kernel never expose "
              "weights; no map is returned rather than a meaningless one.")

    captured: List[torch.Tensor] = []

    def _hook(_m, _inp, out):
        """Capture the (B, L, S) attention weights an MHA block returns."""
        if isinstance(out, (tuple, list)) and len(out) > 1 and \
                isinstance(out[1], torch.Tensor):
            captured.append(out[1].detach())

    handles = [b.register_forward_hook(_hook) for b in blocks]
    try:
        with torch.no_grad():
            wrapped(x)
    finally:
        for h in handles:
            h.remove()

    if not captured:
        raise NoSpatialLayerError(
            f"model_type={model_type or type(model).__name__!r} has "
            f"MultiheadAttention blocks but none returned attention weights "
            f"(they were called with need_weights=False, or the fused kernel "
            f"path was taken). Rollout has nothing to roll; use a gradient or "
            f"perturbation method.")

    fuse = {"mean": lambda a: a.mean(dim=1), "max": lambda a: a.amax(dim=1),
            "min": lambda a: a.amin(dim=1)}
    if head_fusion not in fuse:
        raise AttributionError(
            f"head_fusion must be one of {sorted(fuse)}, got {head_fusion!r}.")

    rolled: Optional[torch.Tensor] = None
    for attn in captured:
        a = attn.double()
        if a.ndim == 4:                      # (B, heads, L, S)
            a = fuse[head_fusion](a)
        a = a[0]                             # (L, S)
        if a.shape[0] != a.shape[1]:
            raise NoSpatialLayerError(
                f"an attention block returned a non-square {tuple(a.shape)} "
                f"matrix, so rollout (which composes square token-to-token "
                f"maps) does not apply to this architecture.")
        if discard_ratio > 0:
            k = int(a.numel() * float(discard_ratio))
            if k > 0:
                flat = a.flatten()
                cut = flat.kthvalue(k).values
                a = torch.where(a <= cut, torch.zeros_like(a), a)
        a = a + torch.eye(a.shape[0], dtype=a.dtype)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rolled = a if rolled is None else a @ rolled

    n_tokens = int(rolled.shape[0])
    grid = int(round(math.sqrt(n_tokens - 1)))
    if grid * grid == n_tokens - 1:
        weights = rolled[0, 1:]              # class token -> patches
    else:
        grid = int(round(math.sqrt(n_tokens)))
        if grid * grid != n_tokens:
            raise NoSpatialLayerError(
                f"{n_tokens} attention tokens do not form a square patch grid "
                f"(with or without a class token), so they cannot be laid back "
                f"out over the image.")
        weights = rolled.mean(dim=0)
    heat = weights.reshape(grid, grid).float()
    return Attribution(
        method="attention_rollout",
        map=_finite_2d(heat, (int(x.shape[-2]), int(x.shape[-1]))),
        target=int(target), n_classes=wrapped.n_classes,
        single_logit=bool(wrapped.single_logit), predicted=int(predicted),
        raw=None, layer=None, family="attention", backend="spacr",
        params={"head_fusion": head_fusion, "discard_ratio": discard_ratio},
        notes=[f"Rolled out {len(captured)} attention blocks.",
               "Attention rollout is not class-conditional: the map is "
               "identical for every target, so it cannot show that the model "
               "distinguished your classes."])


def _attention_adapter(spec: "MethodSpec", wrapped: ClassScoreModel,
                       x: torch.Tensor, target: int, layer: Optional[str],
                       model_type: Optional[str], **kw):
    """Registry entry point for rollout, returning the adapter tuple."""
    att = attention_rollout(wrapped.model, x, target=target,
                            model_type=model_type,
                            head_fusion=str(kw.get("head_fusion", "mean")),
                            discard_ratio=float(kw.get("discard_ratio", 0.0)))
    return att.map, None, None, list(att.notes)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    """One registered attribution method.

    :ivar name: the key callers pass as ``method=``.
    :ivar family: ``'cam'``, ``'gradient'``, ``'perturbation'`` or
        ``'attention'`` — the families fail in different ways, which is why
        agreement *across* families is worth more than agreement within one.
    :ivar backend: ``'torchcam'``, ``'captum'`` or ``'spacr'``.
    :ivar needs_layer: whether a spatial target layer is required.
    :ivar smoothed: whether the adapter should wrap itself in SmoothGrad.
    :ivar description: one line the GUI can show.
    """

    name: str
    family: str
    backend: str
    fn: Callable[..., Any]
    needs_layer: bool = False
    smoothed: bool = False
    description: str = ""

    def smoothgrad_variant(self) -> "MethodSpec":
        """This method with SmoothGrad averaging turned on."""
        return MethodSpec(name=self.name, family=self.family,
                          backend=self.backend, fn=self.fn,
                          needs_layer=self.needs_layer, smoothed=True,
                          description=self.description)


def _spec(name, family, backend, fn, needs_layer=False, description=""):
    """Build one registry entry."""
    return MethodSpec(name=name, family=family, backend=backend, fn=fn,
                      needs_layer=needs_layer, description=description)


ATTRIBUTION_METHODS: Dict[str, MethodSpec] = {
    # -- CAM family: a weighted sum of one conv layer's feature maps.
    "gradcam": _spec("gradcam", "cam", "torchcam", _torchcam_cam, True,
                     "gradient-weighted feature maps; the default"),
    "gradcam_pp": _spec("gradcam_pp", "cam", "torchcam", _torchcam_cam, True,
                        "Grad-CAM++: higher-order weights, better for several "
                        "instances of the same object"),
    "scorecam": _spec("scorecam", "cam", "torchcam", _torchcam_cam, True,
                      "Score-CAM: gradient-free, one forward pass per channel "
                      "— slow but immune to gradient saturation"),
    "xgradcam": _spec("xgradcam", "cam", "torchcam", _torchcam_cam, True,
                      "XGrad-CAM: axiom-derived weights, close to Grad-CAM on "
                      "ReLU CNNs"),
    "layercam": _spec("layercam", "cam", "torchcam", _torchcam_cam, True,
                      "Layer-CAM: per-element weights, usable at earlier "
                      "layers where Grad-CAM degenerates"),
    "eigencam": _spec("eigencam", "cam", "spacr", _eigen_cam, True,
                      "Eigen-CAM: first principal component of the "
                      "activations; class-agnostic, useful as a control"),
    # -- Gradient family: derivative of the class score w.r.t. the input.
    "saliency": _spec("saliency", "gradient", "captum", _captum_attribute,
                      False, "absolute input gradient; the original saliency "
                             "map"),
    "integrated_gradients": _spec(
        "integrated_gradients", "gradient", "captum", _captum_attribute, False,
        "integrated gradients: path integral from a baseline, so it depends on "
        "the baseline you pick"),
    "guided_backprop": _spec(
        "guided_backprop", "gradient", "captum", _captum_attribute, False,
        "guided backpropagation; sharp, and the usual failure case of the "
        "randomisation sanity check"),
    "input_x_gradient": _spec(
        "input_x_gradient", "gradient", "captum", _captum_attribute, False,
        "input times gradient: a first-order Taylor term"),
    "deeplift": _spec("deeplift", "gradient", "captum", _captum_attribute,
                      False, "DeepLIFT rescale rule against a baseline"),
    # -- Perturbation family: the only family that does not need gradients to
    #    be meaningful. Slow, model-agnostic, and the closest thing here to a
    #    direct measurement of what the model uses.
    "occlusion": _spec("occlusion", "perturbation", "captum",
                       _captum_attribute, False,
                       "slide a blanking window over the image and record the "
                       "score drop"),
    "feature_ablation": _spec(
        "feature_ablation", "perturbation", "captum", _captum_attribute, False,
        "blank one tile at a time and record the score drop"),
    # -- Transformers.
    "attention_rollout": _spec(
        "attention_rollout", "attention", "spacr", _attention_adapter, False,
        "attention rollout for transformer backbones with no convolution to "
        "hook; not class-conditional"),
}


def list_methods(family: Optional[str] = None) -> List[str]:
    """Registered method names, optionally restricted to one family."""
    return sorted(n for n, s in ATTRIBUTION_METHODS.items()
                  if family is None or s.family == family)


def methods_by_family() -> Dict[str, List[str]]:
    """Method names grouped by family, families in a stable order."""
    out: Dict[str, List[str]] = {}
    for name in sorted(ATTRIBUTION_METHODS):
        out.setdefault(ATTRIBUTION_METHODS[name].family, []).append(name)
    return out


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def attribute(model: nn.Module, image: Any, method: str = "gradcam", *,
              target: Optional[int] = None, layer: Optional[str] = None,
              model_type: Optional[str] = None, **kw) -> Attribution:
    """Attribute one image with one method.

    Every registered method returns the same thing: a finite ``(H, W)`` map at
    the input's resolution, for the class ``target`` names, for either head
    shape. What differs is what the number means, which is why
    :class:`Attribution` carries the family and the notes.

    :param model: the trained classifier. Left untouched — the wrapper and the
        hooks are removed before this returns.
    :param image: one image, ``(H, W)``, ``(C, H, W)`` or ``(1, C, H, W)``.
    :param method: a key of :data:`ATTRIBUTION_METHODS`.
    :param target: class index to explain; defaults to the model's prediction.
        For a single-logit head, 0 and 1 are both valid and give opposite maps.
    :param layer: dotted target-layer name for the CAM family; defaults to the
        last convolution.
    :param model_type: architecture name, used only to make errors readable.
    :param kw: method-specific options — ``n_steps``/``baseline`` (integrated
        gradients, DeepLIFT), ``window``/``stride`` (occlusion), ``block``
        (feature ablation), ``head_fusion``/``discard_ratio`` (rollout),
        ``n_samples``/``sigma`` when called through :func:`smoothgrad`.
    :returns: the :class:`Attribution`.
    :raises UnknownMethodError: for an unregistered method name.
    :raises NoSpatialLayerError: when a CAM is asked of a model with no
        convolutional feature map, or rollout of a model with no attention.
    :raises AttributionError: for a bad target, layer or baseline.
    """
    spec = ATTRIBUTION_METHODS.get(str(method))
    if spec is None:
        grouped = ", ".join(f"{fam}: {names}"
                            for fam, names in methods_by_family().items())
        raise UnknownMethodError(
            f"unknown attribution method {method!r}. Registered methods by "
            f"family — {grouped}.")
    return _attribute_with_spec(spec, model, image, target=target, layer=layer,
                                model_type=model_type, **kw)


def _attribute_with_spec(spec: MethodSpec, model: nn.Module, image: Any, *,
                         target: Optional[int] = None,
                         layer: Optional[str] = None,
                         model_type: Optional[str] = None,
                         **kw) -> Attribution:
    """Shared body of :func:`attribute` and :func:`smoothgrad`."""
    x = _to_batch(image)
    was_training = model.training
    model.eval()
    wrapped = ClassScoreModel(model)
    try:
        target = _resolve_target(wrapped, x, target)
        predicted = _predicted_class(wrapped, x)
        out = spec.fn(spec, wrapped, x, int(target), layer, model_type, **kw)
    finally:
        if was_training:
            model.train()
    amap, raw, used_layer, notes = out
    amap = np.asarray(amap, dtype=np.float32)
    if not np.isfinite(amap).all():
        amap = np.nan_to_num(amap, nan=0.0, posinf=0.0, neginf=0.0)
    result = Attribution(
        method=spec.name, map=amap, target=int(target),
        n_classes=wrapped.n_classes, single_logit=bool(wrapped.single_logit),
        predicted=int(predicted), raw=raw, layer=used_layer,
        family=spec.family, backend=spec.backend,
        params={"layer": used_layer, "smoothgrad": spec.smoothed, **dict(kw)},
        notes=list(notes))
    if wrapped.single_logit:
        result.notes.append(
            "This model has a single-logit binary head; it was attributed "
            f"through the [-z, +z] two-class view, so target={target} means "
            f"class {target} and not 'the logit'.")
    if result.is_flat():
        result.notes.append(
            "The map is completely flat, so it ranks no pixel above another. "
            "Nothing downstream — deletion, insertion, pointing game — can say "
            "anything about it. For a CAM this usually means the target layer "
            "collapsed to 1x1 or its ReLU suppressed everything.")
    return result


def smoothgrad(model: nn.Module, image: Any, base_method: str = "saliency", *,
               n_samples: int = 25, sigma: float = 0.15,
               target: Optional[int] = None, layer: Optional[str] = None,
               model_type: Optional[str] = None, seed: Optional[int] = None,
               **kw) -> Attribution:
    """SmoothGrad: average ``base_method`` over ``n_samples`` noisy copies.

    Gradient maps are visually noisy because the gradient of a ReLU network
    fluctuates sharply between neighbouring inputs. Averaging over Gaussian
    perturbations of the input suppresses that fluctuation as ``1/sqrt(n)``
    while keeping the structure that survives perturbation.

    For the captum-backed methods this is captum's own ``NoiseTunnel``. The CAM
    family and rollout are not captum attributors, so for those the averaging is
    done here over re-runs of the adapter — same definition, applied to the map.

    Smoothing makes a map *look* better. It does not make it more faithful, and
    a smoothed map that still fails :func:`randomization_sanity_check` fails it
    just as badly.

    :param model: the trained classifier.
    :param image: one image.
    :param base_method: any key of :data:`ATTRIBUTION_METHODS`.
    :param n_samples: number of noisy copies to average. ``1`` is a single
        noisy sample, *not* the clean map — call :func:`attribute` for that.
    :param sigma: noise standard deviation as a fraction of the input's dynamic
        range (the SmoothGrad paper's ``stdev_spread``).
    :param target: class to explain; resolved once on the clean image so the
        noise cannot flip which class is being explained sample to sample.
    :param layer: CAM target layer.
    :param model_type: architecture name for the error messages.
    :param seed: torch seed, so a repeated call reproduces.
    :param kw: forwarded to the base method.
    :returns: the averaged :class:`Attribution`.
    :raises AttributionError: for ``n_samples`` below 1.
    """
    spec = ATTRIBUTION_METHODS.get(str(base_method))
    if spec is None:
        raise UnknownMethodError(
            f"unknown base method {base_method!r} for SmoothGrad; registered "
            f"methods: {list_methods()}.")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise AttributionError(
            f"n_samples must be at least 1, got {n_samples}; SmoothGrad over "
            f"zero samples has nothing to average.")
    if seed is not None:
        torch.manual_seed(int(seed))

    x = _to_batch(image)
    wrapped = ClassScoreModel(model)
    target = _resolve_target(wrapped, x, target)

    if spec.backend == "captum":
        result = _attribute_with_spec(
            spec.smoothgrad_variant(), model, x, target=target, layer=layer,
            model_type=model_type, n_samples=n_samples, sigma=sigma, **kw)
        result.notes.append(
            f"SmoothGrad via captum NoiseTunnel: {n_samples} samples at "
            f"sigma={sigma} of the input range.")
        return result

    stdev = _sigma_to_stdev(sigma, x)
    maps: List[np.ndarray] = []
    template: Optional[Attribution] = None
    for _ in range(n_samples):
        # Every sample is noisy, including the only one when n_samples == 1 —
        # that is what captum's NoiseTunnel does, and the two paths must not
        # disagree about what "SmoothGrad with one sample" means.
        noisy = x + torch.randn_like(x) * stdev
        one = _attribute_with_spec(spec, model, noisy, target=target,
                                   layer=layer, model_type=model_type, **kw)
        maps.append(one.map)
        template = one
    averaged = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    assert template is not None
    template.map = averaged
    template.raw = None
    template.params.update({"smoothgrad": True, "n_samples": n_samples,
                            "sigma": sigma})
    template.notes.append(
        f"SmoothGrad by averaging {n_samples} runs of {spec.name} at "
        f"sigma={sigma} of the input range (this family is not a captum "
        f"attributor, so NoiseTunnel does not apply).")
    return template


def compare_methods(model: nn.Module, image: Any,
                    methods: Sequence[str] = (), *,
                    target: Optional[int] = None, layer: Optional[str] = None,
                    model_type: Optional[str] = None,
                    skip_failures: bool = True,
                    **kw) -> List[Attribution]:
    """Attribute one image with several methods, for side-by-side reading.

    The deliverable is the panel plus :func:`method_agreement` over it, not any
    single map. Methods that cannot run on this architecture (a CAM on a pure
    transformer) are skipped with their reason recorded rather than aborting the
    comparison, unless ``skip_failures`` is off.

    :param model: the trained classifier.
    :param image: one image.
    :param methods: method names; defaults to one representative of each family
        that works on any architecture, plus Grad-CAM.
    :param target: class to explain; resolved once so every method explains the
        same class.
    :param layer: CAM target layer.
    :param model_type: architecture name for the error messages.
    :param skip_failures: record and skip a failing method instead of raising.
    :param kw: forwarded to every method.
    :returns: the attributions, in the order requested. Failures appear as
        :class:`Attribution` objects with a flat map and the error in ``notes``
        only when ``skip_failures`` is True.
    """
    names = list(methods) or ["gradcam", "saliency", "integrated_gradients",
                              "occlusion"]
    x = _to_batch(image)
    wrapped = ClassScoreModel(model)
    target = _resolve_target(wrapped, x, target)
    out: List[Attribution] = []
    for name in names:
        try:
            out.append(attribute(model, x, name, target=target, layer=layer,
                                 model_type=model_type, **kw))
        except Exception as exc:
            if not skip_failures:
                raise
            spec = ATTRIBUTION_METHODS.get(name)
            out.append(Attribution(
                method=name,
                map=np.zeros((int(x.shape[-2]), int(x.shape[-1])),
                             dtype=np.float32),
                target=int(target), n_classes=wrapped.n_classes,
                single_logit=bool(wrapped.single_logit),
                predicted=_predicted_class(wrapped, x),
                family=spec.family if spec else "", layer=layer,
                backend=spec.backend if spec else "",
                notes=[f"FAILED: {type(exc).__name__}: {exc}",
                       "This map is all zeros because the method did not run; "
                       "it is a placeholder, not a result."]))
    return out


# ---------------------------------------------------------------------------
# Analysis 1 — deletion and insertion curves
# ---------------------------------------------------------------------------

@dataclass
class Curve:
    """A deletion or insertion curve and its area.

    :ivar kind: ``'deletion'`` or ``'insertion'``.
    :ivar fractions: fraction of pixels removed / inserted at each step.
    :ivar scores: the target class's probability at each step.
    :ivar auc: area under the curve, trapezoidal over ``fractions``. Bounded in
        ``[0, 1]`` because the scores are probabilities.
    :ivar baseline: what removed pixels were replaced with.
    :ivar target: the class whose probability was tracked.
    :ivar notes: caveats.
    """

    kind: str
    fractions: np.ndarray
    scores: np.ndarray
    auc: float
    baseline: str
    target: int
    notes: List[str] = field(default_factory=list)

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger AUC is the better outcome for this curve's kind."""
        return self.kind == "insertion"

    @property
    def drop(self) -> float:
        """How far the score fell (deletion) or rose (insertion), start to end."""
        return float(self.scores[0] - self.scores[-1])


def _ranked_pixels(amap: np.ndarray) -> np.ndarray:
    """Flat pixel indices ordered by the map, highest first, ties by position."""
    flat = np.asarray(amap, dtype=np.float64).ravel()
    return np.argsort(-flat, kind="stable")


def _perturbation_curve(model: nn.Module, image: Any, amap: Any, kind: str, *,
                        target: Optional[int] = None, n_steps: int = 20,
                        baseline: Any = "blur") -> Curve:
    """Shared body of :func:`deletion_curve` and :func:`insertion_curve`."""
    if kind not in ("deletion", "insertion"):
        raise AttributionError(
            f"kind must be 'deletion' or 'insertion', got {kind!r}.")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise AttributionError(
            f"n_steps must be at least 1, got {n_steps}; a curve needs at "
            f"least one perturbed point besides the unperturbed one.")

    x = _to_batch(image)
    amap = np.asarray(
        amap.map if isinstance(amap, Attribution) else amap, dtype=np.float64)
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if amap.shape != (h, w):
        raise AttributionError(
            f"the attribution map is {amap.shape} but the image is {(h, w)}; "
            f"the curve removes pixels by rank, so they must line up.")

    wrapped = ClassScoreModel(model)
    was_training = model.training
    model.eval()
    try:
        target = _resolve_target(wrapped, x, target)
        ref = _captum_baseline(baseline, x)
        order = _ranked_pixels(amap)
        n_pixels = order.size
        counts = [int(round(n_pixels * i / n_steps)) for i in range(n_steps + 1)]

        fractions: List[float] = []
        scores: List[float] = []
        for k in counts:
            mask = torch.ones(n_pixels, dtype=x.dtype)
            if k:
                mask[torch.as_tensor(order[:k].copy(), dtype=torch.long)] = 0.0
            mask = mask.reshape(1, 1, h, w)
            if kind == "deletion":
                probe = x * mask + ref * (1.0 - mask)
            else:
                probe = ref * mask + x * (1.0 - mask)
            probs = class_scores(wrapped, probe, probability=True)
            fractions.append(k / float(n_pixels))
            scores.append(float(probs[0, int(target)]))
    finally:
        if was_training:
            model.train()

    frac = np.asarray(fractions, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    auc = float(np.trapz(sc, frac)) if frac.size > 1 else float(sc[0])
    label = (str(baseline) if isinstance(baseline, (str, int, float))
             else "custom tensor")
    return Curve(kind=kind, fractions=frac, scores=sc, auc=auc,
                 baseline=label, target=int(target),
                 notes=[CRITERION_CAVEATS[f"{kind}_auc"]])


def deletion_curve(model: nn.Module, image: Any, amap: Any, *,
                   target: Optional[int] = None, n_steps: int = 20,
                   baseline: Any = "blur") -> Curve:
    """Remove the highest-ranked pixels first and track the class probability.

    A faithful map removes the pixels the model actually uses, so the
    probability collapses early and the area under the curve is small. **A flat
    deletion curve is the finding**: the map ranked pixels the model does not
    use, whatever the picture looked like.

    :param model: the trained classifier.
    :param image: the image the map explains.
    :param amap: an :class:`Attribution` or a raw ``(H, W)`` map.
    :param target: class whose probability is tracked; defaults to the
        prediction on the unperturbed image.
    :param n_steps: perturbation steps between 0 % and 100 % removed.
    :param baseline: what removed pixels become — ``'blur'`` (default, the
        least out-of-distribution), ``'zero'``, ``'mean'``, ``'uniform'``, a
        number or a tensor.
    :returns: the :class:`Curve`; lower ``auc`` is better.
    """
    return _perturbation_curve(model, image, amap, "deletion", target=target,
                               n_steps=n_steps, baseline=baseline)


def insertion_curve(model: nn.Module, image: Any, amap: Any, *,
                    target: Optional[int] = None, n_steps: int = 20,
                    baseline: Any = "blur") -> Curve:
    """Start from a blanked image and add the highest-ranked pixels first.

    The mirror of :func:`deletion_curve`, and it answers a different question:
    deletion asks whether the map found pixels that are *necessary*, insertion
    whether it found pixels that are *sufficient*. The two routinely rank
    methods differently, and that disagreement is information, not an error.

    :param model: the trained classifier.
    :param image: the image the map explains.
    :param amap: an :class:`Attribution` or a raw ``(H, W)`` map.
    :param target: class whose probability is tracked.
    :param n_steps: insertion steps between 0 % and 100 % inserted.
    :param baseline: what the not-yet-inserted pixels are.
    :returns: the :class:`Curve`; higher ``auc`` is better.
    """
    return _perturbation_curve(model, image, amap, "insertion", target=target,
                               n_steps=n_steps, baseline=baseline)


def faithfulness(model: nn.Module, image: Any, amap: Any, *,
                 target: Optional[int] = None, n_steps: int = 20,
                 baseline: Any = "blur",
                 mask: Optional[Any] = None) -> Dict[str, Any]:
    """Every faithfulness number for one map, with the caveats attached.

    :param model: the trained classifier.
    :param image: the image the map explains.
    :param amap: an :class:`Attribution` or a raw ``(H, W)`` map.
    :param target: class to score.
    :param n_steps: steps for both curves.
    :param baseline: removal baseline for both curves.
    :param mask: optional boolean object mask enabling the pointing game.
    :returns: dict with ``deletion_auc``, ``insertion_auc``, ``deletion`` and
        ``insertion`` :class:`Curve` objects, ``pointing_game`` (or None),
        ``flat`` and ``notes``.
    """
    dele = deletion_curve(model, image, amap, target=target, n_steps=n_steps,
                          baseline=baseline)
    ins = insertion_curve(model, image, amap, target=target, n_steps=n_steps,
                          baseline=baseline)
    raw_map = np.asarray(amap.map if isinstance(amap, Attribution) else amap)
    out: Dict[str, Any] = {
        "deletion": dele, "insertion": ins,
        "deletion_auc": dele.auc, "insertion_auc": ins.auc,
        "pointing_game": None if mask is None else pointing_game(raw_map, mask),
        "flat": bool(float(raw_map.max() - raw_map.min()) <= 0),
        "notes": [NOT_AN_EXPLANATION,
                  CRITERION_CAVEATS["deletion_auc"],
                  CRITERION_CAVEATS["insertion_auc"]],
    }
    if out["flat"]:
        out["notes"].append(
            "The map is flat, so both curves describe removing pixels in "
            "arbitrary order. Neither AUC means anything here.")
    if mask is not None:
        out["notes"].append(CRITERION_CAVEATS["pointing_game"])
    return out


# ---------------------------------------------------------------------------
# Analysis 2 — the pointing game against spaCR's own object masks
# ---------------------------------------------------------------------------

def pointing_game(amap: Any, mask: Any, *, tolerance: int = 0) -> float:
    """Does the map's brightest pixel land inside the object?

    spaCR already has the answer key: ``merged/*.npy`` stores the label-mask
    planes next to the image channels, so a boolean object mask is free. The
    game is deliberately crude — one pixel, hit or miss — because that is all it
    claims to measure.

    :param amap: an :class:`Attribution` or a ``(H, W)`` map.
    :param mask: object mask, same spatial shape. Any non-zero value is inside
        the object, so a spaCR integer label plane can be passed directly.
    :param tolerance: dilate the mask by this many pixels before testing, the
        allowance the original pointing-game protocol uses for maps computed at
        a coarser resolution than the image.
    :returns: 1.0 for a hit, 0.0 for a miss.
    :raises AttributionError: on a shape mismatch or an empty mask — an empty
        mask would score 0.0 and look like a method failure rather than a
        missing annotation.
    """
    m = np.asarray(amap.map if isinstance(amap, Attribution) else amap,
                   dtype=np.float64)
    obj = np.asarray(mask)
    if obj.ndim > 2:
        obj = obj.reshape(-1, *obj.shape[-2:]).any(axis=0)
    obj = obj != 0
    if obj.shape != m.shape:
        raise AttributionError(
            f"the object mask is {obj.shape} but the attribution map is "
            f"{m.shape}; the pointing game compares one to the other pixel for "
            f"pixel.")
    if not obj.any():
        raise AttributionError(
            "the object mask is empty, so there is nothing for the map to "
            "point at. A score of 0 here would say the method failed when in "
            "fact the annotation is missing.")
    if int(tolerance) > 0:
        pad = int(tolerance)
        grown = np.zeros_like(obj)
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                grown |= np.roll(np.roll(obj, dy, axis=0), dx, axis=1)
        obj = grown
    row, col = divmod(int(np.argmax(m)), m.shape[1])
    return 1.0 if bool(obj[row, col]) else 0.0


def pointing_game_rate(maps: Sequence[Any], masks: Sequence[Any], *,
                       tolerance: int = 0) -> Dict[str, Any]:
    """Pointing-game hit rate over a set of images.

    :param maps: attributions or raw maps.
    :param masks: the matching object masks.
    :param tolerance: passed to :func:`pointing_game`.
    :returns: dict with ``rate``, ``hits``, ``n``, ``skipped`` (images whose
        mask was empty or mismatched, which are excluded rather than counted as
        misses) and ``notes``.
    :raises AttributionError: when the two sequences differ in length.
    """
    if len(maps) != len(masks):
        raise AttributionError(
            f"got {len(maps)} maps and {len(masks)} masks; the pointing game "
            f"needs one mask per map.")
    hits = 0
    scored = 0
    skipped: List[str] = []
    for i, (m, k) in enumerate(zip(maps, masks)):
        try:
            hits += int(pointing_game(m, k, tolerance=tolerance))
            scored += 1
        except AttributionError as exc:
            skipped.append(f"image {i}: {exc}")
    notes = [CRITERION_CAVEATS["pointing_game"]]
    if skipped:
        notes.append(
            f"{len(skipped)} of {len(maps)} images were excluded (not counted "
            f"as misses) because their mask could not be used: {skipped[:3]}")
    if scored == 0:
        notes.append(
            "No image could be scored, so the rate below is not a measurement.")
    return {"rate": (hits / scored) if scored else float("nan"),
            "hits": hits, "n": scored, "skipped": skipped, "notes": notes}


# ---------------------------------------------------------------------------
# Analysis 3 — the model-randomisation sanity check (Adebayo et al. 2018)
# ---------------------------------------------------------------------------

@dataclass
class SanityCheck:
    """Result of randomising the model's weights and attributing again.

    :ivar method: the method under test.
    :ivar mode: ``'cascading'`` (randomise layers from the output backwards,
        accumulating) or ``'independent'`` (one layer at a time, from a fresh
        copy).
    :ivar stages: ``(layer_name, similarity)`` after each randomisation step, in
        the order applied.
    :ivar final_similarity: similarity to the trained model's map once every
        parameterised layer has been randomised. This is the number that
        matters: at that point the model is noise.
    :ivar max_similarity: the largest similarity over all stages.
    :ivar threshold: the value ``final_similarity`` must fall below to pass.
    :ivar passed: whether the method's map changed when the weights did.
    :ivar metric: name of the similarity measure.
    :ivar notes: the verdict in words.
    """

    method: str
    mode: str
    stages: List[Tuple[str, float]]
    final_similarity: float
    max_similarity: float
    threshold: float
    passed: bool
    metric: str = "spearman_abs"
    notes: List[str] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """``1 - final_similarity``, clipped to ``[0, 1]``: higher is better.

        This is the form the hyperparameter search ranks on, so that every
        criterion there points the same way.
        """
        if not math.isfinite(self.final_similarity):
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - self.final_similarity)))

    def verdict(self) -> str:
        """One sentence a user can act on."""
        if self.passed:
            return (f"{self.method} PASSES the randomisation sanity check: "
                    f"after randomising every layer its map correlates "
                    f"{self.final_similarity:.2f} with the trained model's "
                    f"(threshold {self.threshold:.2f}), so it depends on the "
                    f"weights.")
        return (f"{self.method} FAILS the randomisation sanity check: with "
                f"every weight randomised its map still correlates "
                f"{self.final_similarity:.2f} with the trained model's "
                f"(threshold {self.threshold:.2f}). It is describing the image, "
                f"not the model — do not read it as an explanation of this "
                f"classifier.")


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two maps' absolute values.

    Rank-based on purpose: attribution scales are arbitrary and differ between
    methods, so only the ordering of pixels is comparable. Returns NaN when
    either map is constant, because a constant map has no ordering to correlate.
    """
    x = np.abs(np.asarray(a, dtype=np.float64)).ravel()
    y = np.abs(np.asarray(b, dtype=np.float64)).ravel()
    if x.size != y.size:
        raise AttributionError(
            f"cannot correlate maps of different sizes ({x.size} vs {y.size}).")
    if x.size < 2 or float(x.max() - x.min()) <= 0 or float(y.max() - y.min()) <= 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(x, y).statistic)
    except Exception:
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho if math.isfinite(rho) else float("nan")


def _parameterised_modules(model: nn.Module) -> List[str]:
    """Names of modules owning parameters directly, in definition order."""
    return [name for name, mod in model.named_modules()
            if name and any(True for _ in mod.parameters(recurse=False))]


def _randomize_module(module: nn.Module, generator: torch.Generator) -> None:
    """Replace a module's own parameters with noise of the same scale.

    Matching the original scale matters: parameters drawn far outside the
    trained range saturate the network, every logit collapses to the same value,
    and the resulting map is flat for *every* method — which would make every
    method appear to pass.
    """
    with torch.no_grad():
        for param in module.parameters(recurse=False):
            std = float(param.detach().float().std())
            if not math.isfinite(std) or std <= 0:
                std = 0.05
            noise = torch.randn(param.shape, generator=generator,
                                dtype=torch.float32).to(param.dtype)
            param.copy_(noise * std)


def randomization_sanity_check(
        model: nn.Module, image: Any, method: Union[str, Callable] = "gradcam",
        *, target: Optional[int] = None, layer: Optional[str] = None,
        model_type: Optional[str] = None, mode: str = "cascading",
        threshold: float = 0.5, seed: int = 0,
        max_stages: Optional[int] = None,
        attribute_fn: Optional[Callable[..., Any]] = None,
        **kw) -> SanityCheck:
    """Adebayo et al. 2018: does the map change when the weights are destroyed?

    Randomise the model's parameters layer by layer, from the output backwards,
    re-attribute at every stage, and correlate each map with the map from the
    trained model. A method that depends on what the model learned produces an
    uncorrelated map once the weights are noise. Several widely used methods do
    not: guided backprop and guided Grad-CAM are the canonical failures, and a
    method that fails this is an edge detector being read as an explanation.

    **This is the most informative check in the module.** A map that passes
    deletion and insertion but fails this one is describing the image; a method
    that fails here cannot be rescued by smoothing, a better colormap or a
    different target layer.

    :param model: the trained classifier. Deep-copied — the original is never
        modified.
    :param image: one image.
    :param method: a registered method name, or any callable when
        ``attribute_fn`` is not given.
    :param target: class to explain, resolved once against the trained model and
        held fixed, so a stage's map is not silently for a different class.
    :param layer: CAM target layer.
    :param model_type: architecture name for the error messages.
    :param mode: ``'cascading'`` (default, the paper's) or ``'independent'``.
    :param threshold: rank correlation below which the method passes.
    :param seed: RNG seed for the randomisation, so the check reproduces.
    :param max_stages: cap on the number of layers randomised, from the output
        backwards. The final stage always randomises everything regardless.
    :param attribute_fn: ``fn(model, image, target=...) -> map or Attribution``,
        for testing a method that is not in the registry.
    :param kw: forwarded to the attribution call.
    :returns: the :class:`SanityCheck`.
    :raises AttributionError: for an unknown mode, or a model with no
        parameters to randomise.
    """
    if mode not in ("cascading", "independent"):
        raise AttributionError(
            f"mode must be 'cascading' or 'independent', got {mode!r}.")

    x = _to_batch(image)
    wrapped = ClassScoreModel(model)
    target = _resolve_target(wrapped, x, target)
    method_name = method if isinstance(method, str) else getattr(
        method, "__name__", "custom")

    def _run(m: nn.Module) -> np.ndarray:
        """Attribute ``image`` with the method under test on model ``m``."""
        if attribute_fn is not None:
            out = attribute_fn(m, x, target=int(target))
        elif callable(method) and not isinstance(method, str):
            out = method(m, x, target=int(target))
        else:
            out = attribute(m, x, str(method), target=int(target), layer=layer,
                            model_type=model_type, **kw)
        return np.asarray(out.map if isinstance(out, Attribution) else out,
                          dtype=np.float64)

    reference = _run(model)

    names = _parameterised_modules(model)
    if not names:
        raise AttributionError(
            "the model has no parameterised layers, so there is nothing to "
            "randomise and the sanity check cannot be run.")
    order = list(reversed(names))
    if max_stages is not None and int(max_stages) > 0:
        order = order[:int(max_stages)]

    generator = torch.Generator().manual_seed(int(seed))
    stages: List[Tuple[str, float]] = []
    cascade = copy.deepcopy(model)
    cascade.eval()
    for name in order:
        if mode == "cascading":
            probe = cascade
        else:
            probe = copy.deepcopy(model)
            probe.eval()
        _randomize_module(dict(probe.named_modules())[name], generator)
        stages.append((name, _rank_correlation(reference, _run(probe))))

    # The verdict is taken from a model in which EVERY layer is noise, whatever
    # max_stages capped the reported stages at. A method judged on a partly
    # randomised model gets an easy pass. Its generator is seeded separately
    # from the stage loop's, so the verdict does not depend on how many stages
    # happened to run before it — otherwise the same seed gives two different
    # answers for max_stages=1 and max_stages=None.
    full_generator = torch.Generator().manual_seed(int(seed) + 991)
    full = copy.deepcopy(model)
    full.eval()
    for name in reversed(names):
        _randomize_module(dict(full.named_modules())[name], full_generator)
    final = _rank_correlation(reference, _run(full))
    if not stages or stages[-1][0] != names[0] or mode != "cascading":
        stages.append(("<all layers>", final))

    finite = [s for _n, s in stages if math.isfinite(s)]
    max_sim = max(finite) if finite else float("nan")
    passed = bool(math.isfinite(final) and final < float(threshold))

    notes = [CRITERION_CAVEATS["sanity_gap"]]
    if not math.isfinite(final):
        notes.append(
            "The randomised model produced a constant map, so no rank "
            "correlation exists and the check is inconclusive rather than "
            "passed. A flat map is not evidence of sensitivity to the weights.")
        passed = False
    result = SanityCheck(method=method_name, mode=mode, stages=stages,
                         final_similarity=final, max_similarity=max_sim,
                         threshold=float(threshold), passed=passed, notes=notes)
    result.notes.insert(0, result.verdict())
    return result


# ---------------------------------------------------------------------------
# Analysis 4 — agreement between methods
# ---------------------------------------------------------------------------

@dataclass
class Agreement:
    """Pairwise rank correlation between several methods on the same image.

    :ivar methods: method names, in matrix order.
    :ivar matrix: symmetric ``(n, n)`` Spearman correlation matrix.
    :ivar mean: mean of the off-diagonal entries.
    :ivar minimum: smallest off-diagonal entry.
    :ivar pairs: every ``(method_a, method_b, rho)``, most disagreeing first.
    :ivar notes: the verdict in words.
    """

    methods: List[str]
    matrix: np.ndarray
    mean: float
    minimum: float
    pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def verdict(self) -> str:
        """One sentence, deliberately asymmetric — see the note in the source."""
        if not math.isfinite(self.mean):
            return ("Agreement could not be computed: at least one map is "
                    "constant and has no pixel ordering to correlate.")
        if self.minimum < 0.2:
            return (f"The methods DISAGREE (lowest pair rho={self.minimum:.2f}, "
                    f"mean {self.mean:.2f}). They cannot all be describing what "
                    f"this model uses, so no single map here should be trusted "
                    f"on its own.")
        if self.mean > 0.7:
            return (f"The methods agree (mean rho={self.mean:.2f}). Agreement "
                    f"is weak evidence: methods sharing a failure mode — the "
                    f"gradient family shares several — agree with each other "
                    f"while all being wrong. Check the randomisation sanity "
                    f"check before reading agreement as confirmation.")
        return (f"The methods partly agree (mean rho={self.mean:.2f}, lowest "
                f"pair {self.minimum:.2f}). Read the panel, not the ranking.")


def method_agreement(attributions: Sequence[Any]) -> Agreement:
    """Rank correlation between several attribution maps of the same image.

    Agreement and disagreement are not symmetric evidence. Methods agreeing is
    weak: the gradient family shares failure modes, so its members agree with
    each other whether or not any of them is faithful. Methods disagreeing is
    strong: at most one of them can be right, so none should be quoted alone.

    :param attributions: :class:`Attribution` objects or raw ``(H, W)`` maps;
        at least two, all the same shape.
    :returns: the :class:`Agreement`.
    :raises AttributionError: for fewer than two maps or a shape mismatch.
    """
    if len(attributions) < 2:
        raise AttributionError(
            f"agreement needs at least two maps to compare, got "
            f"{len(attributions)}.")
    names: List[str] = []
    maps: List[np.ndarray] = []
    for i, a in enumerate(attributions):
        if isinstance(a, Attribution):
            names.append(a.method)
            maps.append(np.asarray(a.map, dtype=np.float64))
        else:
            names.append(f"map_{i}")
            maps.append(np.asarray(a, dtype=np.float64))
    shapes = {m.shape for m in maps}
    if len(shapes) > 1:
        raise AttributionError(
            f"the maps have different shapes {sorted(shapes)}; agreement "
            f"compares them pixel for pixel.")

    n = len(maps)
    matrix = np.eye(n, dtype=np.float64)
    pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = _rank_correlation(maps[i], maps[j])
            matrix[i, j] = matrix[j, i] = rho
            pairs.append((names[i], names[j], rho))
    off = [p[2] for p in pairs if math.isfinite(p[2])]
    mean = float(np.mean(off)) if off else float("nan")
    minimum = float(np.min(off)) if off else float("nan")
    pairs.sort(key=lambda p: (math.inf if not math.isfinite(p[2]) else p[2]))
    result = Agreement(methods=names, matrix=matrix, mean=mean,
                       minimum=minimum, pairs=pairs,
                       notes=[NOT_AN_EXPLANATION])
    result.notes.insert(0, result.verdict())
    if len(off) < len(pairs):
        result.notes.append(
            f"{len(pairs) - len(off)} pairs could not be correlated because a "
            f"map is constant; they are excluded rather than scored as "
            f"disagreement.")
    return result


# ---------------------------------------------------------------------------
# Drop-in generator for the existing activation-map pipeline
# ---------------------------------------------------------------------------

class AttributionMapGenerator:
    """Batch adapter with the interface ``generate_activation_map`` already uses.

    :class:`spacr.utils.GradCAMGenerator` and
    :class:`spacr.utils.SaliencyMapGenerator` expose
    ``compute_*_and_predictions(X)`` plus ``plot_activation_grid``. This offers
    the same two calls for every method in :data:`ATTRIBUTION_METHODS`, so the
    existing batch loop gains twelve methods without changing shape.

    :param model: the trained classifier.
    :param method: a key of :data:`ATTRIBUTION_METHODS`.
    :param target_layer: CAM target layer, or None for the last convolution.
    :param model_type: architecture name, used in the error messages.
    :param smoothgrad_samples: when above 1, each map is SmoothGrad-averaged.
    :param smoothgrad_sigma: SmoothGrad noise as a fraction of the input range.
    :param kw: forwarded to the method.
    """

    def __init__(self, model, method: str = "gradcam",
                 target_layer: Optional[str] = None,
                 model_type: Optional[str] = None,
                 smoothgrad_samples: int = 0,
                 smoothgrad_sigma: float = 0.15, **kw):
        """Validate the method name up front rather than mid-batch."""
        if str(method) not in ATTRIBUTION_METHODS:
            raise UnknownMethodError(
                f"unknown attribution method {method!r}. Registered methods by "
                f"family — " + ", ".join(
                    f"{fam}: {names}"
                    for fam, names in methods_by_family().items()))
        self.model = model
        self.method = str(method)
        self.target_layer = target_layer
        self.model_type = model_type
        self.smoothgrad_samples = int(smoothgrad_samples or 0)
        self.smoothgrad_sigma = float(smoothgrad_sigma)
        self.kw = dict(kw)
        self.model.eval()

    def compute_maps_and_predictions(self, X):
        """Attribute every image in a batch.

        :param X: batch tensor ``(N, C, H, W)``.
        :returns: ``(maps, predictions)`` — maps is ``(N, H, W)`` float32,
            predictions is ``(N,)`` long, correct for either head shape.
        """
        maps: List[np.ndarray] = []
        preds: List[int] = []
        for i in range(int(X.shape[0])):
            one = X[i:i + 1]
            if self.smoothgrad_samples > 1:
                att = smoothgrad(self.model, one, self.method,
                                 n_samples=self.smoothgrad_samples,
                                 sigma=self.smoothgrad_sigma,
                                 layer=self.target_layer,
                                 model_type=self.model_type, **self.kw)
            else:
                att = attribute(self.model, one, self.method,
                                layer=self.target_layer,
                                model_type=self.model_type, **self.kw)
            maps.append(att.map)
            preds.append(att.predicted)
        return (torch.from_numpy(np.stack(maps, axis=0)),
                torch.tensor(preds, dtype=torch.long))

    # Aliases so this drops into either branch of the existing batch loop.
    compute_gradcam_and_predictions = compute_maps_and_predictions
    compute_saliency_and_predictions = compute_maps_and_predictions

    def plot_activation_grid(self, X, maps, predictions, overlay=True,
                             normalize=False):
        """Render the batch grid, reusing spaCR's existing layout.

        :param X: the input batch.
        :param maps: the attribution maps.
        :param predictions: predicted class per image.
        :param overlay: draw the map over the image.
        :param normalize: percentile-stretch the image under the overlay.
        :returns: the matplotlib Figure.
        """
        from .utils import SaliencyMapGenerator
        return SaliencyMapGenerator(self.model).plot_activation_grid(
            X, maps, predictions, overlay=overlay, normalize=normalize)
