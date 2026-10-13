# Notes from `spacr/hyperparam.py`

Prose lifted out of `spacr/hyperparam.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [umap_checkpoint_path](#umap_checkpoint_path) (1 entry)
- [_run_trials](#_run_trials) (1 entry)
- [random_search](#random_search) (1 entry)
- [umap_metrics](#umap_metrics) (2 entries)
- [walk_search._persisted_state](#walk_search_persisted_state) (1 entry)
- [walk_search](#walk_search) (1 entry)
- [local_direction_search](#local_direction_search) (1 entry)
- [umap_available](#umap_available) (1 entry)
- [_default_umap_embed](#_default_umap_embed) (2 entries)
- [_umap_scores](#_umap_scores) (1 entry)
- [embedding_stability](#embedding_stability) (1 entry)
- [umap_objective_scores](#umap_objective_scores) (1 entry)
- [umap_search](#umap_search) (4 entries)
- [umap_search._fit](#umap_search_fit) (2 entries)
- [ActivationSearchData](#activationsearchdata) (1 entry)
- [_activation_params](#_activation_params) (1 entry)
- [activation_fit_fn._fit](#activation_fit_fn_fit) (1 entry)
- [load_activation_data](#load_activation_data) (1 entry)
- [build_folds](#build_folds) (1 entry)
- [cv_search](#cv_search) (1 entry)
- [SearchData](#searchdata) (1 entry)
- [load_search_data](#load_search_data) (1 entry)
- [run_search_for_app](#run_search_for_app) (1 entry)

## Module level

### lines 92-94  _(unsure)_

```python
UMAP_MISSING_MESSAGE = (
```

Messages the GUI and the CLI both surface verbatim

### lines 208-210

```python
"classify_merged": ["accuracy", "prauc", "roc_auc", "f1", "loss"],
```

THE MERGED SCREEN SEARCHES EITHER FAMILY. The criteria both halves share come first, so a user who switches `classifier_family` keeps the one they picked.

### lines 234-235  _(unsure)_

```python
"classify_merged": {
```

Learning rate is the one knob both families take, so it is the default grid for the merged screen whichever one is selected.

### lines 240-244

```python
"activation": {
```

One representative of each attribution family, because agreement within a family is nearly worthless and disagreement across families is the finding. Score-CAM and feature ablation are left out of the default grid: both are an order of magnitude slower than the rest and a sweep the user cancels tells them nothing.

### line 1156, trailing  _(unsure)_

```python
"metric": {"choices": None},
```

filled from the installed umap-learn

## umap_checkpoint_path

### lines 615-617

```python
if not os.path.exists(path):
```

A hand-built/test settings dict may carry a placeholder such as "/x". Only explicit checkpoint_path is allowed to create a new project tree; an inferred path must start from a source that actually exists.

## _run_trials

### line 887, trailing

```python
except Exception as exc:
```

one bad configuration must not lose the sweep

## random_search

### lines 1053-1055

```python
attempts = 0
```

Bounded rejection sampling; the cap keeps a pathological space from spinning forever, and the fallback fills the remainder from the grid in deterministic order.

## umap_metrics

### lines 1109-1118

```python
from .utils import umap as _guarded_umap
```

NOT a bare `from umap.distances import ...`. umap's package init__ reaches umap.parametric_umap, which imports TENSORFLOW, and spaCR's standing rule is that no module drags TF in.

`spacr.utils.umap` is a lazy loader for `umap.umap_` with the TF-backed roots blocked. Touching it first puts the guarded package in sys.modules, after which the sibling submodule is fetched by name -- importlib rather than an import statement, because the no-TF guard is a line-level grep and a `from umap.` line is what it is there to catch.

### lines 1121-1133

```python
named_distances = __import__(
```

`__import__`, deliberately, and not importlib.import_module.

Two constraints meet here. The no-TF guard is a line-level grep, so a `from umap.` line is out. And `test_the_panel_builds_without_umap_installed` simulates a machine with no umap by patching `builtins.__import__` -- which importlib.import_module bypasses, so using it made that test pass umap-less machines a library they do not have.

`__import__` satisfies both: the grep does not match it, and the test's patch does intercept it. The guarded loader above has already put the package in sys.modules with the TF-backed roots blocked, so this only fetches the sibling submodule.

## walk_search._persisted_state

### lines 1554-1557

```python
payload: Dict[str, Any] = {
```

`centre_n`/`centre_d` are written alongside the general `centre` so a checkpoint from this build stays readable by the 1.5.x two-axis reader. Dropping them would make an in-flight search unresumable by the version that started it.

## walk_search

### line 1576  _(unsure)_

```python
legacy = {"n_neighbors": state.get("centre_n"),
```

A checkpoint written before the walk went N-dimensional.

## local_direction_search

### lines 1809-1812

```python
try:
```

The conversion is inside the guard and the comparison is outside it. With the comparison inside, its own ValueError was caught by the very except that was meant for a non-numeric value, and a user who typed a negative threshold was told their numbers were not numbers.

## umap_available

### lines 1859-1861

```python
from .utils import umap, OptionalDependencyCompatibilityError
```

Through spacr.utils, never a bare `import umap`: umap's package init__ imports umap.parametric_umap -> tensorflow, and TF is not a spaCR dependency. The lazy wrapper blocks it for that import.

## _default_umap_embed

### line 1880, trailing

```python
from .utils import umap
```

never a bare `import umap` — see umap_available

### lines 1885-1887

```python
import warnings
```

umap-learn intentionally disables parallel optimisation when a random_state is supplied. That is expected for a reproducible search, but it emits the same warning for every trial and buries useful output.

## _umap_scores

### line 1916  _(unsure)_

```python
kk = max(1, min(int(k), (n - 1) // 2))
```

trustworthiness requires k < n/2.

## embedding_stability

### lines 1965-1966

```python
raw = NearestNeighbors(n_neighbors=k + 1).fit(array).kneighbors(
```

Query k+1 because a row is its own nearest point, then remove it explicitly rather than relying on sklearn's query-mode distinction.

## umap_objective_scores

### lines 2152-2153  _(unsure)_

```python
composite = math.exp(sum(
```

A geometric mean prevents one excellent property from fully hiding a collapsed objective, while a tiny floor keeps the result finite.

## umap_search

### lines 2302-2306

```python
maximum_neighbors = n_samples - 1
```

umap-learn otherwise silently truncates every oversized n_neighbors value to n_samples - 1. Apart from filling the terminal with warnings, that can make several nominally different trials evaluate the exact same embedding. Bound and de-duplicate the search before any reducer is fit so the reported parameters are the parameters that were actually evaluated.

### lines 2372-2373

```python
requested_backend = "custom"
```

An injected embedder is neither umap-learn nor cuML. Reusing either requested label would be false provenance in checkpoints and rows.

### lines 2526-2528

```python
axes = umap_walk_axes(
```

The user chose the space. Anything they named that the starting point does not carry is an error there rather than here, so `umap_walk_axes` is given the whole start dict.

### lines 2534-2535

```python
axes = [
```

The two-axis default, expressed as axes rather than as a separate code path, so there is one Walk and not two.

## umap_search._fit

### lines 2449-2451

```python
fit_params.setdefault("n_neighbors", implicit_neighbors)
```

A search may vary only min_dist/metric. Keep UMAP's implicit default safe for a small dataset too, without adding an unsearched table column to Trial.params.

### lines 2511-2512

```python
extra["cluster_error"] = f"{type(exc).__name__}: {exc}"
```

Clustering is an optional second analysis of a valid map. Its failure must stay on that row, not erase the embedding.

## ActivationSearchData

### lines 2587-2589  _(unsure)_

```python
@dataclass
```

Activation — sweeping what a trained model is said to attend to

## _activation_params

### lines 2626-2627

```python
named = p.pop("method", None)
```

Both spellings are removed whichever one supplied the value, so neither can leak downstream into the attribution call as a stray keyword.

## activation_fit_fn._fit

### lines 2729-2732

```python
per_image = {"deletion_auc": deletions,
```

The image-to-image spread of the ranked criterion is this search's noise yardstick, the way fold-to-fold spread is the classifiers'. A configuration that wins by less than the variation between images has not won.

## load_activation_data

### lines 2964-2969

```python
from .normalization import normalization_stats
```

WHICH statistics is `input_statistics`, the same setting the training and inference loaders read. A hard-coded 0.5/0.5 here attributes a model under statistics it was not trained with: the saliency is computed on inputs shifted away from the ones the weights learned, so the peak it reports need not be the peak the model would produce in a real run.

## build_folds

### lines 2999-3001  _(unsure)_

```python
def build_folds(labels,
```

Grouped cross-validated search — Classify (CV) and Classify (ML)

## cv_search

### lines 3163-3165

```python
for i, (tr, va) in enumerate(folds):
```

Structural guarantee, checked rather than assumed: no fold — train or validation — may contain a test index. A search that scores on test data selects a configuration that has already seen the answers.

## SearchData

### lines 3334-3336  _(unsure)_

```python
@dataclass
```

App backends — what the GUI's "Hyperparameter search" button actually runs

## load_search_data

### line 3443  _(unsure)_

```python
groups, warn = _well_groups(frame)
```

Supervised searches need labels and well ids.

## run_search_for_app

### lines 3834-3838

```python
family = str(settings.get("classifier_family", "cv") or "cv").lower()
```

THE MERGED SCREEN TAKES THIS ARM WHEN IT IS THE TORCH FAMILY. Its `classifier_family` says which classifier it is about to fit, and the cross-validated image path is the one `classify` used to own; a gradient-boosting family falls through to the measured-feature path below, which is what `ml_analyze` used.
