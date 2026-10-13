# Notes from `spacr/attribution.py`

Prose lifted out of `spacr/attribution.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [ClassScoreModel](#classscoremodel) (1 entry)
- [_eigen_cam](#_eigen_cam) (4 entries)
- [_captum_attribute](#_captum_attribute) (2 entries)
- [attention_rollout](#attention_rollout) (4 entries)
- [smoothgrad](#smoothgrad) (1 entry)
- [pointing_game](#pointing_game) (1 entry)
- [SanityCheck](#sanitycheck) (1 entry)
- [randomization_sanity_check](#randomization_sanity_check) (1 entry)
- [AttributionMapGenerator](#attributionmapgenerator) (2 entries)

## Module level

### line 66

```python
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz
```

np.trapz was removed in numpy 2.0; np.trapezoid is the replacement.

### lines 106-108  _(unsure)_

```python
NOT_AN_EXPLANATION = (
```

Messages surfaced verbatim by the GUI, the CLI and the search

### line 1109  _(unsure)_

```python
"gradcam": _spec("gradcam", "cam", "torchcam", _torchcam_cam, True,
```

CAM family: a weighted sum of one conv layer's feature maps.

### line 1127  _(unsure)_

```python
"saliency": _spec("saliency", "gradient", "captum", _captum_attribute,
```

Gradient family: derivative of the class score w.r.t. the input.

### lines 1144-1146

```python
"occlusion": _spec("occlusion", "perturbation", "captum",
```

Perturbation family: the only family that does not need gradients to be meaningful. Slow, model-agnostic, and the closest thing here to a direct measurement of what the model uses.

## ClassScoreModel

### lines 172-174  _(unsure)_

```python
class ClassScoreModel(nn.Module):
```

Head-shape handling: one logit or C logits, one contract

## _eigen_cam

### line 649, trailing  _(unsure)_

```python
act = captured[0][0]
```

(C, H, W)

### line 651, trailing  _(unsure)_

```python
flat = act.reshape(c, h * w).T
```

(H*W, C)

### lines 653-654  _(unsure)_

```python
try:
```

full_matrices=False keeps this cheap for wide feature maps; a rank-0 (all-constant) activation makes SVD degenerate, so fall back to the mean.

### line 660, trailing  _(unsure)_

```python
if float(proj.sum()) < 0:
```

sign of a singular vector is free

## _captum_attribute

### lines 785-790

```python
import warnings as _warnings
```

captum warns rather than raises for the conditions that silently corrupt a result — most importantly a model that reuses one ReLU instance across layers, which torchvision's ResNets do and which makes DeepLIFT's rescale rule attribute through the wrong activation. A warning printed once to stderr during a batch job is a warning nobody reads, so it is captured and carried on the result instead.

### lines 805-809

```python
if "more than once" in str(exc) or "required for DeepLift" in str(exc):
```

torchvision's ResNets — spaCR's most common backbone after MaxViT — build one `nn.ReLU(inplace=True)` and call it at several points. DeepLIFT's rescale rule needs one hook per activation *use*, so it dies here with a message about module attributes that does not tell a user what to do next.

## attention_rollout

### lines 971-983

```python
restore = _ask_for_attention_weights(blocks)
```

THE BLOCKS ARE ASKED FOR THEIR WEIGHTS, not merely watched.

A hook can only capture what `forward` RETURNS, and torchvision's ViT calls `self.self_attention(x, x, x, need_weights=False)` -- so nothing was ever returned to capture, and this method raised on the only architecture family it exists for. Driven on vit_b_16, which is one of the ten backbones spaCR offers: "has MultiheadAttention blocks but none returned attention weights". The docstring described what the blocks would do if they were asked, which nobody was doing.

`need_weights=True` also takes PyTorch off its fused kernel, which is the point: the fused path computes no explicit attention matrix at all. It is slower and it runs once, under no_grad, for one image.

### line 1010, trailing  _(unsure)_

```python
if a.ndim == 4:
```

(B, heads, L, S)

### line 1012, trailing  _(unsure)_

```python
a = a[0]
```

(L, S)

### line 1031, trailing  _(unsure)_

```python
weights = rolled[0, 1:]
```

class token -> patches

## smoothgrad

### lines 1328-1330

```python
noisy = x + torch.randn_like(x) * stdev
```

Every sample is noisy, including the only one when n_samples == 1 — that is what captum's NoiseTunnel does, and the two paths must not disagree about what "SmoothGrad with one sample" means.

## pointing_game

### lines 1595-1597  _(unsure)_

```python
def pointing_game(amap: Any, mask: Any, *, tolerance: int = 0) -> float:
```

Analysis 2 — the pointing game against spaCR's own object masks

## SanityCheck

### lines 1682-1684

```python
@dataclass
```

Analysis 3 — the model-randomisation sanity check (Adebayo et al. 2018)

## randomization_sanity_check

### lines 1880-1885

```python
full_generator = torch.Generator().manual_seed(int(seed) + 991)
```

The verdict is taken from a model in which EVERY layer is noise, whatever max_stages capped the reported stages at. A method judged on a partly randomised model gets an easy pass. Its generator is seeded separately from the stage loop's, so the verdict does not depend on how many stages happened to run before it — otherwise the same seed gives two different answers for max_stages=1 and max_stages=None.

## AttributionMapGenerator

### lines 2015-2017  _(unsure)_

```python
class AttributionMapGenerator:
```

Drop-in generator for the existing activation-map pipeline

### line 2086  _(unsure)_

```python
compute_gradcam_and_predictions = compute_maps_and_predictions
```

Aliases so this drops into either branch of the existing batch loop.
