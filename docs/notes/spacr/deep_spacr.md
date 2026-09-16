# Notes from `spacr/deep_spacr.py`

Prose lifted out of `spacr/deep_spacr.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (8 entries)
- [autocasting](#autocasting) (1 entry)
- [pick_device](#pick_device) (2 entries)
- [apply_model](#apply_model) (3 entries)
- [apply_model_to_tar](#apply_model_to_tar) (3 entries)
- [_binary_metrics](#_binary_metrics) (4 entries)
- [_multiclass_metrics](#_multiclass_metrics) (10 entries)
- [attach_per_class_columns](#attach_per_class_columns) (1 entry)
- [evaluate_model_performance](#evaluate_model_performance) (10 entries)
- [test_model_core](#test_model_core) (2 entries)
- [test_model_performance](#test_model_performance) (1 entry)
- [resolve_class_balance_loss](#resolve_class_balance_loss) (1 entry)
- [summarize_cv_metrics](#summarize_cv_metrics) (1 entry)
- [_cross_validate_model._fit_one](#_cross_validate_model_fit_one) (1 entry)
- [_cross_validate_model._inner_loader](#_cross_validate_model_inner_loader) (1 entry)
- [_cross_validate_model](#_cross_validate_model) (3 entries)
- [train_test_model](#train_test_model) (9 entries)
- [_plot_training_curves](#_plot_training_curves) (3 entries)
- [format_model_card](#format_model_card) (1 entry)
- [train_model](#train_model) (32 entries)
- [generate_activation_map](#generate_activation_map) (13 entries)
- [visualize_classes](#visualize_classes) (1 entry)
- [visualize_integrated_gradients](#visualize_integrated_gradients) (3 entries)
- [SmoothGrad.compute_smooth_grad](#smoothgradcompute_smooth_grad) (1 entry)
- [visualize_smooth_grad](#visualize_smooth_grad) (2 entries)
- [save_top_class_examples](#save_top_class_examples) (3 entries)
- [deep_spacr](#deep_spacr) (11 entries)
- [model_knowledge_transfer](#model_knowledge_transfer) (13 entries)
- [model_fusion](#model_fusion) (4 entries)
- [model_fusion.combine_tensors](#model_fusioncombine_tensors) (3 entries)
- [annotate_filter_vision.filter_csv_by_png](#annotate_filter_visionfilter_csv_by_png) (5 entries)

## Module level

### lines 25-27

```python
from .errors import RunLedger
```

Fail-loud accounting: a cross-validation fold that dies must not be averaged away silently, and an optional plot that fails must still be visible somewhere other than /dev/null.

### line 29, trailing  _(unsure)_

```python
from .plot import save_figure
```

every kept figure goes through the format/DPI preference

### lines 30-31

```python
from .runctx import resolve_seed, seed_everything, seed_worker, torch_generator
```

One seed reaching Python, NumPy and Torch (CPU + CUDA) rather than only the split helpers. See spacr.runctx.

### lines 38-40

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### line 1074  _(unsure)_

```python
CV_METRIC_KEYS = ('accuracy', 'loss', 'prauc', 'f1_macro',
```

Driven on a three-class dataset built from plate1 of the tsg101 screen.

### line 1974, trailing  _(unsure)_

```python
_TRAIN_CURVE_COLOR = _CLASS_CURVE_COLORS[1]
```

blue

### line 1975, trailing  _(unsure)_

```python
_VAL_CURVE_COLOR = _CLASS_CURVE_COLORS[0]
```

teal

### lines 2161-2163  _(unsure)_

```python
MODEL_CARD_SUFFIX = '.card.json'
```

Model cards — what a checkpoint was trained on, and how well it did

## autocasting

### lines 197-201

```python
dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
```

CPU autocast accepted float16 only in newer Torch releases.  The supported 2.1 floor accepts bfloat16 there; CUDA uses float16 in every supported release. Production enables AMP only for CUDA, but keeping this helper valid for either device makes its context-manager contract independently testable on CPU-only hosts.

## pick_device

### lines 242-248

```python
return found.torch_device, ""
```

ANY OTHER ACCELERATOR IS TAKEN AT FACE VALUE. The free-memory check below is `torch.cuda.mem_get_info`, which exists only on CUDA -- Metal shares memory with the system and has no equivalent, and asking ROCm costs a context for a number spaCR would only use to print. A real OOM stays a real failure, which is the same bargain the missing-mem_get_info branch already strikes for old CUDA drivers.

### lines 253-254

```python
return torch.device("cuda"), ""
```

An older driver with no mem_get_info. Use the card and let a real OOM be a real failure; guessing would be worse.

## apply_model

### lines 345-350

```python
from .normalization import normalization_stats
```

WHICH statistics is now a setting. spaCR has always used 0.5/0.5, which maps [0,1] to [-1,1]; every ImageNet-pretrained torchvision model was fitted on 0.485/0.456/0.406 and 0.229/0.224/0.225, so a finetune under the old default hands pretrained weights inputs distributed differently from the ones they learned. The default is unchanged so existing scores do not move under anybody.

### lines 367-372

```python
dataset = NoClassDataset(data_dir=src, transform=transform, shuffle=False,
```

`len(dataset)`, NOT `len(src)`. Both of these counted the CHARACTERS IN THE PATH: a run over a folder whose name happened to be 98 characters long announced "Loading dataset ... with 98 images" and then "Loaded 98 images", and returned an empty frame. The number was plausible, it was printed twice, and it had nothing to do with the data (236 B5).

### lines 377-379

```python
raise ValueError(
```

AND AN EMPTY FOLDER IS NOT A RESULT. It returned a frame with the right columns and no rows, which reads downstream as "the model scored nothing" rather than "there was nothing to score".

## apply_model_to_tar

### lines 464-465  _(unsure)_

```python
from .normalization import describe_normalization, normalization_stats
```

See the note on the other transform: which statistics is a setting now, and the model card records the answer.

### lines 495-499

```python
if getattr(dataset, 'crop_format', None) is not None:
```

A tar built from on-demand crops carries the crop-format marker, so say which channel ordering the model is about to be shown. The pixels are NOT re-ordered here: a model's weights are tied to the order it was trained on, and quietly correcting a legacy archive at inference time would invalidate every model trained before spaCR grew the marker.

### lines 566-567  _(unsure)_

```python
df['cv_predictions'] = df['predicted_label'].astype(int)
```

Multiclass predictions are class indices, not a binary threshold on the winning class's confidence.

## _binary_metrics

### line 598

```python
thresholds = np.append(thresholds, 1.0)
```

F1-optimal threshold (optional; we still report 0.5 preds below)

### line 608

```python
pred = (pos_probs >= 0.5).astype(int)
```

Discrete preds at 0.5 threshold for stability/readability

### lines 624-625

```python
"f1_macro": (float(f1_score(y_true, pred, average='macro',
```

train_model has always PRINTED f1_macro, but neither metric helper returned it, so it read nan on every line and never reached the CSV.

### lines 629-634

```python
"per_class_accuracy": [
```

Binary reported its two class accuracies under names nothing else understood, so every consumer that wanted "the per-class numbers" had to branch on the head shape. Report the same two values under the SAME key the multiclass path uses, so the live view, the TensorBoard scalars and the model card are one code path. neg/pos stay for backwards compatibility.

## _multiclass_metrics

### lines 652-655

```python
return {
```

scikit-learn 1.7 rejects empty arrays in confusion_matrix. An empty validation split is still a valid evaluator result: its metrics are undefined, its class schema is known, and no fabricated sample should be introduced merely to make a dependency accept the call.

### lines 673-678

```python
row_sums = cm.sum(axis=1)
```

The old `cm.sum(axis=1, where=(rowsums != 0), initial=1)` looked like a divide-by-zero guard but was neither: `initial` seeds np.add.reduce, so it added 1 to *every* row sum (a perfect classifier scored diag/(rowsum+1)), and the (C,) mask broadcasts over the LAST axis of the (C, C) matrix, so it dropped columns instead of rows. Guard the row sums explicitly; classes with no true support report 0.0.

### lines 681-682  _(unsure)_

```python
y_true_oh = np.zeros((len(y_true), C), dtype=int)
```

Average precision macro (one-vs-rest)

Build one-hot y_true

### lines 688-689  _(unsure)_

```python
logging.getLogger('spacr.deep_spacr').error(
```

NaN is written straight into the metrics CSV, where it is indistinguishable from "not computed". Say why, at least once.

### line 696  _(unsure)_

```python
return {
```

For compatibility with your logging keys:

### line 699, trailing  _(unsure)_

```python
"neg_accuracy": np.nan,
```

not meaningful in multiclass

### line 700, trailing  _(unsure)_

```python
"pos_accuracy": np.nan,
```

not meaningful in multiclass

### line 701, trailing  _(unsure)_

```python
"prauc": float(ap_macro),
```

reuse key for macro-AP

### lines 703-705

```python
"f1_macro": (float(f1_score(y_true, preds, average='macro',
```

Macro F1 is the metric that actually matters on an imbalanced screen: accuracy is dominated by the majority class, and this weights every class equally. It was printed but never computed.

### lines 710-713

```python
"class_support": [int(v) for v in row_sums],
```

Support belongs beside the accuracy it was computed from: a class at 0.40 over 500 objects is a broken classifier, the same 0.40 over 5 objects is two mistakes. Without it, the per-class line invites exactly the wrong reading.

## attach_per_class_columns

### lines 785-787

```python
metrics['class_names'] = [name for name, _, _ in rows]
```

Stamped so the history is self-describing: everything downstream (the live plot, the model card) can name the classes from one epoch dict rather than needing the list threaded through it.

## evaluate_model_performance

### line 852, trailing  _(unsure)_

```python
head_dim = None
```

infer from first batch

### line 860  _(unsure)_

```python
if head_dim is None:
```

infer head size/mode once

### line 867  _(unsure)_

```python
target = target.to(device).float()
```

BCE-style targets: float {0,1}, allow (N,) or (N,1)

### line 871  _(unsure)_

```python
if target.ndim == 2:
```

CE-style: class indices (N,)

### line 873  _(unsure)_

```python
target = target.argmax(dim=1)
```

handle one-hot inputs robustly

### line 878  _(unsure)_

```python
local_loss_fn = loss_fn
```

choose loss (prefer training's loss_fn if provided)

### line 881  _(unsure)_

```python
local_loss_fn = build_loss(loss_type or 'auto',
```

fallback: construct something reasonable matching the head

### line 905  _(unsure)_

```python
mean_loss = total_loss / max(1, total_samples)
```

aggregate

### line 910  _(unsure)_

```python
if (num_classes or head_dim or 1) == 1:
```

empty loader: synthesize empty array with correct rank

### line 919  _(unsure)_

```python
if probs_np.ndim == 1:
```

metrics (assumes _binary_metrics / _multiclass_metrics exist)

## test_model_core

### line 996, trailing  _(unsure)_

```python
probs_rows.append(probs.reshape(-1, 1))
```

keep 2D for uniform handling

### line 1015  _(unsure)_

```python
df_dict = {
```

Build per-file results dataframe

## test_model_performance

### line 1060  _(unsure)_

```python
result_df = pd.DataFrame([data_dict])
```

The old function returned a DataFrame in 'result'; emulate that:

## resolve_class_balance_loss

### lines 1110-1111  _(unsure)_

```python
return loss_type, (
```

Both corrections multiply: the rare class ends up over-weighted and the model swings to over-predicting it.

## summarize_cv_metrics

### line 1154  _(unsure)_

```python
std = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
```

ddof=1: folds are a sample of the possible splits, not the population.

## _cross_validate_model._fit_one

### lines 1300-1303

```python
gradient_accumulation=int(
```

DERIVED, NOT STORED. `steps = 1` is the off state, so the step count alone says whether to accumulate -- see `settings._fold_gradient_accumulation` for why the boolean that used to sit beside it was folded in.

## _cross_validate_model._inner_loader

### lines 1355-1358

```python
return DataLoader(
```

A shuffled loader with no generator draws its permutation from torch's global RNG, and a worker inherits (fork) or loses (spawn) the parent's stream -- so the inner folds were never reproducible even with random_seed set. See spacr.runctx.seed_worker.

## _cross_validate_model

### lines 1377-1378

```python
ledger = RunLedger('cross_validation')
```

A fold that does not train is dropped from the spread. Two dead folds out of five used to produce a "5-fold CV" summary computed on three.

### lines 1520-1525

```python
attach_per_class_columns(metrics, settings.get('classes'))
```

FLATTENED FIRST. The per-class accuracies live in `metrics` as a LIST under 'per_class_accuracy', which no spread statistic can aggregate; `attach_per_class_columns` is what turns them into one scalar column per class, and it had only ever been called on the way to the epoch CSVs. So a cross-validation over three classes reported accuracy, loss and prauc and nothing per class.

### lines 1601-1603

```python
ledger.finalize(artifact=folds_loc, threshold=0.5)
```

The per-fold CSV is stamped so a reader can see it covers fewer folds than requested, and a run in which most folds died aborts outright: the "spread" of two surviving folds out of five is not a spread.

## train_test_model

### lines 1674-1680

```python
seed_everything(resolve_seed(settings),
```

random_seed used to reach the split helpers below and nothing else: torch's own initialisation -- weight init, dropout, the shuffle inside every DataLoader -- was never seeded at all, so two "identical" runs trained two different models. One call fixes Python, NumPy and Torch (CPU and CUDA); what it still cannot promise is in spacr.runctx.SeedReport.caveats, and cudnn.benchmark (set True at the top of this module) is one of the things deterministic=True undoes.

### lines 1702-1704

```python
if settings.get('leakage_audit_train_test', True):
```

Audit the permanent dataset boundary before a model sees a pixel. This catches renamed byte-identical copies as well as plate/well/object and exported-augmentation relationships.

### lines 1734-1737

```python
class_balance = settings.get('class_balance', 'none')
```

Class-imbalance steering: 'weighted_loss' is expressed as a loss_type, the sampler modes as a DataLoader sampler inside generate_loaders. Either way the change is announced before the settings snapshot is written, so the saved settings record what actually ran.

### lines 1757-1760

```python
if settings['train'] and settings['test']:
```

This ladder used to sit inside an outer `if settings['train']:`, which made the test-only arm unreachable (a test-only run snapshotted nothing), and the `is True` comparisons also skipped the snapshot for truthy-but-not-True flags such as train=1 coming from a scripted caller.

### lines 1768-1773

```python
try:
```

save_settings writes to <src>/settings/<name>.csv, and the name is keyed on model_type and epochs alone -- so a second run of the same shape with a different learning rate silently OVERWRITES the first run's snapshot, and the first run's curves become unattributable. A copy inside dst is per-run by construction, since dst already varies with the run. spacr.train_compare.load_run prefers this one.

### lines 1788-1790

```python
cv_result_loc = _cross_validate_model(settings, num_classes)
```

k-fold replaces the single split entirely: every crop is validated once, and the reported number is a mean with its fold-to-fold spread rather than one draw from it.

### lines 1901-1903

```python
print(f"Training aborted: model_type {settings['model_type']!r} could not be built.")
```

choose_model could not build model_type (e.g. a typo in settings). Abort here rather than falling through into the test branch, where pick_best_model would look for a checkpoint that was never written.

### lines 1925-1926  _(unsure)_

```python
print(f'Loading selected checkpoint for testing: {model_path}')
```

Test the checkpoint selected by validation, not the final in-memory epoch (which may already have overfit).

### lines 1955-1956  _(unsure)_

```python
return cv_result_loc if cv_folds >= 2 else model_path
```

In k-fold mode there is no single "the model"; the per-fold metric CSV is the artefact worth handing back.

## _plot_training_curves

### lines 2039-2040

```python
class_hist = val_hist if val_hist else train_hist
```

Prefer held-out per-class accuracy; fall back to train so a run without a validation split still gets the panel rather than a blank third.

### lines 2046-2049

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 2053-2057

```python
fig.patch.set_alpha(0.0)
```

Transparent from the start, so the container shows through and the page opacity reaches the plot. The GUI restyles text and spines for the active theme when it renders (figure_queue._style_figure_colors); what matters here is that no opaque page is baked in, because a white or black rectangle cannot be undone by restyling.

## format_model_card

### lines 2421-2423

```python
for note in held.get('notes') or []:
```

Anything held_out_report had to say about WHICH rows those are. A card whose n and whose accuracy describe different populations is the failure this block exists to make impossible to miss.

## train_model

### line 2581  _(unsure)_

```python
early_stopping_patience=0,  # 0 = disabled; e.g. 20 = stop after 20 epochs without val improvement
```

add early stopping parameters

### line 2582, trailing

```python
early_stopping_patience=0,
```

0 = disabled; e.g. 20 = stop after 20 epochs without val improvement

### lines 2690-2691  _(unsure)_

```python
classes = sorted([d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_...
```

The folder names in ImageFolder's sorted order ARE the head order, so they win over whatever the caller passed.

### lines 2694-2698

```python
classes = None
```

...but with no folder tree to read (a tar-backed dataset, a caller supplying its own loaders), the caller's list is the only class naming there is. The old `else: classes = None` threw it away, so the checkpoint and every per-class report came out as class_0/class_1 even when the names were passed in.

### lines 2719-2722

```python
resume_payload = None
```

NO `if model is None` BRANCH. `choose_model` raises now, naming the setting, the value it was given and the nearest spellings -- which is what "Model X not found" followed by (None, None) and a failure three frames later never managed to say. See instruction 236 B4.

### lines 2793-2794

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
```

`verbose` was deprecated in torch 2.2 and removed in 2.5; passing it made this documented schedule raise TypeError before the first batch.

### line 2798  _(unsure)_

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
```

FIX: new option — cosine annealing

### lines 2833-2834

```python
live_train_hist, live_val_hist = [], []
```

Full per-epoch history kept for the live training plot (the accumulators above get consumed/cleared by _save_progress each epoch).

### lines 2837-2840

```python
held_out_raw = None
```

(epoch, metrics, [probs, labels]) of the epoch whose weights became the best checkpoint — the ONLY held-out evaluation that describes the file the model card is written beside. Using the last epoch's numbers for a checkpoint saved five epochs earlier is the quiet way a card lies.

### lines 2842-2843

```python
_curve_ledger = RunLedger('train_model:live_curves')
```

Kept separate from any training ledger: a failed live plot says nothing about whether the weights are trustworthy.

### lines 2847-2849

```python
mixed_precision, amp_note = resolve_mixed_precision(
```

MIXED PRECISION, asked for by `amp` and only ever taken on a card that has tensor cores. `mixed_precision` is the answer for THIS machine, so everything below is one code path rather than two.

### line 2870  _(unsure)_

```python
n_batches = len(train_loaders)
```

record total number of batches so we can detect leftover gradients

### lines 2875-2878

```python
with autocasting(mixed_precision, device):
```

HALF PRECISION FOR THE FORWARD AND THE LOSS, full precision for the weights. See `autocasting`: on a card with tensor cores this is most of the speed and half the activation memory, and outside one it is a no-op context.

### lines 2899-2901

```python
scaler.scale(loss).backward()
```

SCALED, or float16 gradients underflow to zero and the model simply does not learn. The scaler is a no-op when it is disabled, so this is one code path rather than two.

### line 2909  _(unsure)_

```python
if gradient_accumulation and (n_batches % gradient_accumulation_steps != 0):
```

flush leftover accumulated gradients at the end of the epoch

### line 2915  _(unsure)_

```python
train_time = time.time() - start_time
```

Epoch end: evaluate

### lines 2925-2927

```python
train_dict['lr'] = float(optimizer.param_groups[0]['lr'])
```

The schedule moves the learning rate every epoch and nothing recorded it, so "why did the curve bend at epoch 30" was unanswerable from the run folder alone.

### line 2932  _(unsure)_

```python
val_dict = None
```

initialize val_dict to None so the variable always exists for _save_model

### lines 2955-2957

```python
class_line = format_per_class_accuracy(val_dict, classes, 'Val ')
```

The aggregate above is the number that hides a dead class. Print the breakdown on its own line, every epoch, held-out first — not once at the end, by which point the run is over.

### line 2962  _(unsure)_

```python
current_val_acc = val_dict.get('accuracy', 0.0)
```

track best validation accuracy for early stopping and best-model selection

### lines 2998-3004

```python
live_train_hist.append(train_dict)
```

Live training curves — follow loss/accuracy in real time in the GUI when plot is enabled. Each epoch refreshes the same figure (the GUI bridge captures plt.show and routes it to the figure view). Accumulated unconditionally: the accumulators above are consumed and cleared by _save_progress every epoch, so this is the only in-memory record of the run, and the model card's curve needs it whether or not anyone asked for a live plot.

### lines 3009-3011

```python
with _curve_ledger.item(f'epoch_{epoch}', stage='live_curves'):
```

Cosmetic: a live curve that fails to render must not kill the training run. It must not be *invisible* either — the bare `pass` here hid a broken plot for the whole run.

### lines 3013-3015

```python
live_figure = _plot_training_curves(
```

Class names ride along inside the epoch dicts (see attach_per_class_columns), so this call site keeps the signature every existing caller and stub already has.

### line 3022  _(unsure)_

```python
scheduler.step()
```

FIX: also step cosine scheduler here

### line 3025  _(unsure)_

```python
if accumulated_val_dicts:
```

Save rolling CSVs

### line 3032  _(unsure)_

```python
will_stop = (
```

pass val_dict to _save_model so checkpoint decisions use validation accuracy

### line 3050  _(unsure)_

```python
if model_path is not None and is_best:
```

track the best model path based on validation accuracy

### line 3056  _(unsure)_

```python
if will_stop:
```

early stopping — break if val hasn't improved for `patience` epochs

### line 3062

```python
if (epoch % 25 == 0) or (epoch == epochs):
```

Periodic suggestions (every 25 epochs and final epoch)

### lines 3077-3079

```python
_curve_ledger.finalize()
```

Not stamped and not fatal — the training artifacts are unaffected — but a run where every live plot failed now says so instead of ending with a silently empty figure pane.

### line 3084

```python
final_path = best_model_path if best_model_path is not None else model_path
```

return best_model_path if available, otherwise fall back to last model_path

### lines 3088-3091

```python
try:
```

A card that fails to write must not lose the weights that were just trained for six hours, but it must also not fail silently — an uncarded checkpoint that nobody noticed is the state this feature exists to end.

## generate_activation_map

### lines 3172-3178

```python
_LEGACY_CAM_TYPES = ('gradcam', 'gradcam_pp', 'saliency_image',
```

Anything outside the four legacy names is one of the methods registered in spacr.attribution (Grad-CAM++, Score-CAM, XGrad-CAM, Layer-CAM, Eigen-CAM, guided backprop, input x gradient, DeepLIFT, integrated gradients, occlusion, feature ablation, attention rollout). They run through the same batch loop via AttributionMapGenerator, which exposes the same compute_*_and_predictions / plot_activation_grid calls the two legacy generators do.

### line 3192  _(unsure)_

```python
n_jobs = settings['n_jobs']
```

Set number of jobs for loading

### lines 3197-3200

```python
transform_steps = [
```

Set transforms for images. The Normalize step has to be appended conditionally: an inline `... if normalize_input else None` put a literal None into the Compose list, so normalize_input=False raised "TypeError: 'NoneType' object is not callable" on the first image.

### lines 3206-3208

```python
from .normalization import normalization_stats
```

`normalize_input` stays the on/off it has always been; WHICH statistics is `input_statistics`, so an existing settings file keeps its meaning exactly.

### line 3220  _(unsure)_

```python
if not os.path.exists(settings['dataset']):
```

Handle dataset path

### line 3230  _(unsure)_

```python
dataset_dir = os.path.dirname(settings['dataset'])
```

Create directory for saving activation maps if it does not exist

### lines 3246-3247

```python
data_loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=settings['shuffle'],...
```

Seeded generator + worker init: which images land in the activation-map batches is otherwise a different sample every run.

### line 3252  _(unsure)_

```python
if use_attribution:
```

Initialize generator based on cam_type

### lines 3306-3310

```python
if use_attribution or settings['cam_type'] in ['saliency_image', 'gradcam', 'gradcam_pp']:
```

A flat map (e.g. a Grad-CAM fully suppressed by its F.relu, which happens whenever the target layer has collapsed to 1x1) has max == min, so the unguarded min-max rescale below used to produce 0/0 -> all-NaN and then an undefined NaN -> uint8 cast. `rng > 0` is also False for NaN, so a map that arrives already NaN is absorbed too.

### lines 3312-3315

```python
lo = activation_map.min()
```

Every spacr.attribution method returns a single (H, W) map, so it takes the same greyscale path the summed saliency and the CAMs already took. activation_map = activation_map.sum(axis=0)

### line 3323  _(unsure)_

```python
rgb_activation_map = np.zeros((activation_map.shape[1], activation_map.shape[2], 3), dtype=np.uint8)
```

Handle each channel separately and save as RGB

### line 3325, trailing  _(unsure)_

```python
for c in range(min(activation_map.shape[0], 3)):
```

Limit to 3 channels for RGB

### line 3333  _(unsure)_

```python
class_pred = predicted_classes[i].item()
```

Save activation maps

## visualize_classes

### line 3506, trailing  _(unsure)_

```python
for target_y in range(2):
```

Assuming binary classification

## visualize_integrated_gradients

### lines 3560-3563

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 3572-3574

```python
overlay = np.array(image.resize((image_size, image_size)))
```

Same trap as in visualize_smooth_grad: `image` is the unresized original while the attribution map is image_size square, so the blend below only broadcast when the source PNG happened to be image_size square.

### line 3577, trailing  _(unsure)_

```python
integrated_grads_rgb = np.stack([integrated_grads] * 3, axis=-1)
```

Convert saliency map to RGB

## SmoothGrad.compute_smooth_grad

### lines 3622-3625

```python
output[:, target_class].sum().backward()
```

Back-propagate the whole target column, not just row 0: with

`output[0, target_class]` autograd only populated row 0 of .grad, so a batched input silently came back with all-zero attributions for every sample after the first. Identical for a single sample.

## visualize_smooth_grad

### lines 3675-3678

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 3687-3691

```python
overlay = np.array(image.resize((image_size, image_size)))
```

preprocess_image returns the UNRESIZED PIL image next to the resized tensor, so blending np.array(image) with the image_size-sized map raised a broadcast ValueError for any source PNG that is not image_size square. Blend at the resolution the model actually saw (a no-op copy when they already match); ax[0] still shows the full-resolution original.

## save_top_class_examples

### lines 3743-3744  _(unsure)_

```python
selections = []
```

Build each folder's selection once. ``classes`` contains human-readable labels, whereas the probability-column suffix is the model-output index.

### lines 3769-3770  _(unsure)_

```python
member_destinations = {}
```

Build a lookup: tar member name → list of destination paths. An image can legitimately appear at both binary extremes in a one-row result.

### line 3782  _(unsure)_

```python
extracted = 0
```

single pass through the tar: extract only the members we need

## deep_spacr

### line 3899

```python
from .settings import deep_spacr_defaults
```

local imports kept inside to avoid import cycles on some setups

### line 3904  _(unsure)_

```python
settings = deep_spacr_defaults(settings)
```

1) expand defaults (now supports things like metadata_rules, annotation_columns, etc.)

### line 3908  _(unsure)_

```python
save_settings(settings, name='DL_model')
```

persist a snapshot of the config for reproducibility

### line 3925, trailing  _(unsure)_

```python
return
```

or raise RuntimeError if you prefer hard fail

### line 3927  _(unsure)_

```python
settings['src'] = os.path.dirname(train_path)
```

point training to the newly created train folder by default

### line 3949  _(unsure)_

```python
settings['src'] = src_before
```

restore original src (so later steps like apply can use the user’s dataset if needed)

### lines 3952-3954

```python
tar_path = settings.get('tar_path')
```

3) build the full, unlabelled inference dataset independently of model application when requested. Applying a model still creates it on demand, preserving the previous one-switch workflow.

### line 3969  _(unsure)_

```python
if settings.get('apply_model_to_dataset'):
```

4) apply model to the full dataset/tar

### line 3974  _(unsure)_

```python
_flowview_advance("evaluation")
```

run inference and get the results DataFrame

### lines 3979-3980  _(unsure)_

```python
examples_dst = os.path.join(os.path.dirname(tar_path), 'top_examples')
```

NEW: save the top-N most confident images per class dst sits next to the tar file, in a subfolder called 'top_examples'

### lines 3987-3988  _(unsure)_

```python
_flowview_advance("scores")
```

NEW: merge predictions back into the measurements database settings['src'] can be a string or list; use the first entry

## model_knowledge_transfer

### line 4039  _(unsure)_

```python
if student_save_path.endswith('.pth'):
```

Adjust filename to reflect knowledge-distillation if desired

### line 4082  _(unsure)_

```python
optimizer = optim.Adam(student_model.parameters(), lr=lr)
```

You could load a partial checkpoint into the student here if desired.

### line 4087  _(unsure)_

```python
for epoch in range(epochs):
```

Distillation training loop

### line 4098  _(unsure)_

```python
logits_s = student_model(images)         # shape: (B, num_classes)
```

Forward pass student

### line 4099, trailing  _(unsure)_

```python
logits_s = student_model(images)
```

shape: (B, num_classes)

### line 4100, trailing  _(unsure)_

```python
logits_s_temp = logits_s / temperature
```

scale by T

### line 4102  _(unsure)_

```python
with torch.no_grad():
```

Distillation from teachers

### line 4104  _(unsure)_

```python
teacher_probs_list = []
```

We'll average teacher probabilities

### line 4114  _(unsure)_

```python
teacher_probs_ensemble = torch.mean(torch.stack(teacher_probs_list), dim=0)
```

average them

### line 4117  _(unsure)_

```python
if num_classes == 1:
```

Student probabilities (log-softmax)

### line 4125  _(unsure)_

```python
loss_distill = F.kl_div(
```

Distillation loss => KLDiv

### lines 4132-4133  _(unsure)_

```python
if num_classes == 1:
```

Real label loss => cross-entropy

We can compute this on the raw logits or scaled. Typically raw logits is standard:

### line 4140  _(unsure)_

```python
loss = alpha * loss_ce + (1 - alpha) * loss_distill
```

Weighted sum

## model_fusion

### line 4193  _(unsure)_

```python
print(f"Loading the first model from: {model_paths[0]} to derive architecture")
```

1. Load the first checkpoint to figure out architecture & hyperparams

### line 4205  _(unsure)_

```python
for path in model_paths[1:]:
```

2. Load the rest of the checkpoints

### line 4263  _(unsure)_

```python
all_tensors = [sd[key] for sd in state_dicts]
```

gather all versions of this tensor

### line 4267  _(unsure)_

```python
fused_model.load_state_dict(fused_sd)
```

Load combined weights into the fused model

## model_fusion.combine_tensors

### line 4227  _(unsure)_

```python
first = tensor_list[0]
```

stack along new dimension => shape (num_models, *tensor.shape)

### lines 4230-4232

```python
return first.clone()
```

Counters such as BatchNorm.num_batches_tracked are state, not learnable weights. Combining them numerically corrupts their meaning, so retain the first compatible model's value.

### lines 4241-4242  _(unsure)_

```python
zero = (stacked == 0).any(dim=0)
```

Neural weights are signed. Use a signed geometric mean of magnitudes and preserve the sign of the arithmetic mean.

## annotate_filter_vision.filter_csv_by_png

### lines 4305-4308

```python
marker = os.sep + "datasets" + os.sep
```

Split the path to identify the datasets folder and build the training folder path. Unpacking the split into two names raised a bare "not enough values to unpack" for any CSV outside a '.../datasets/...' tree; say what is wrong instead, since remove_train cannot locate the training images without it.

### line 4322  _(unsure)_

```python
df = pd.read_csv(csv_file)
```

Load the CSV file into a DataFrame

### line 4325  _(unsure)_

```python
png_files = set()
```

Collect PNG filenames from train/nc and train/pc

### line 4328, trailing  _(unsure)_

```python
if os.path.exists(folder):
```

Ensure the folder exists

### line 4331  _(unsure)_

```python
filtered_df = df[~df['path'].isin(png_files)]
```

Filter the DataFrame by excluding rows where filenames match PNG files
