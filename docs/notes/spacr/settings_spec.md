# Notes from `spacr/settings_spec.py`

Prose lifted out of `spacr/settings_spec.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [_value_special_cases](#_value_special_cases) (1 entry)
- [convert_settings_dict_for_gui](#convert_settings_dict_for_gui) (14 entries)

## Module level

### lines 24-27

```python
_TORCHVISION_MODELS_CURATED = [
```

Curated torchvision classification models for the `model_type` combo. Kept static so opening a settings screen never triggers a slow `import torchvision`. The pipeline validates/instantiates the real model by name at train time.

### lines 149-153

```python
(('both', 'grna', 'gene'),
```

NAMED FOR WHAT THEY PRODUCE. 'grna'/'gene'/'both' are the keys the settings file and the API have always used and they do not change; what changes is that the panel says which analysis each one asks for, so a reader who wants gene effects can find them without knowing that `level` is the control that gives them.

## _value_special_cases

### lines 184-185

```python
return (kind, list(options), current)
```

The panel's own value is the default, so opening a settings screen never rewrites the setting it was opened on.

## convert_settings_dict_for_gui

### lines 202-208

```python
torchvision_models = _torchvision_model_names()
```

NOTE: we deliberately do NOT `import torchvision` here. Enumerating the torchvision model zoo pulls in torch + torchvision, a ~5 s import that made every FIRST module open sluggish. The classify pipeline still instantiates the real torchvision model by name at train time — the GUI combo just needs a list of valid names, so we use a curated static list (if torchvision happens to be imported already we extend it with the full zoo, for free).

### lines 210-213

```python
cellpose_models = _cellpose_model_names()
```

Same bargain, for the same measured reason: `cellpose.models` pulls in torch (~2.5 s) and this runs while a settings page is being built, so the accessor reads the API only when Cellpose is already loaded and degrades to the shipped list otherwise. It is never empty.

### lines 219-226

```python
'analysis_mode': ('combo',
```

Instruction 134: two valid values, and it was a free-text box in both front ends. Declared here rather than only in the Qt combo table so the two GUIs cannot offer different lists. THE LABELS READ, and the VALUES do not change (134's third point). 'guide_permutation' is what the settings key is called; what the dropdown shows is the sentence, the same way 132's model box explains what it fits. (value, label) pairs, so every settings file already written goes on meaning what it meant.

### lines 235-245

```python
'grna_statistic': ('combo', ['pearson', 'rank'], 'pearson'),
```

Instruction 135, and the same argument as `analysis_mode` above: two valid values, and the RUN now has to agree with the volcano's right-click menu about which P value 'significant' meant. A free-text box lets a settings CSV say 'Adjusted' or 'bh' and be refused at the seam instead of picked from a list of two. Declared here rather than only in the Qt combo table so the Tk and Qt panels cannot offer different lists. OFFERED, NOT TYPED. Two spellings, and a free-text box lets a settings CSV say 'spearman' -- a reasonable guess for what 'rank' does -- and be refused at the seam rather than picked from a list of two.

### lines 252-254

```python
'dataset_mode': ('combo', ['annotation', 'metadata'], 'metadata'),
```

io.generate_training_dataset dispatches on metadata|annotation| measurement and returns (None, None) for anything else. 'recruitment' was offered here and silently produced no dataset.

### lines 267-283

```python
'regression_type': ('combo', _regression_type_choices(), 'mixed'),
```

DEFAULT 'mixed' since 2026-08-17, matching settings.get_perform_regression_default_settings: "mixed answers the most central question best". A combo whose default differs from the settings default posts a different model than the one the panel was built for. READ FROM THE INVENTORY, NOT LISTED BY HAND. The hand-written list offered 'gls' -- which is in UNSUPPORTED_REGRESSION_TYPES and RAISES -- and omitted six families that fit: huber, beta, quasi_binomial, elasticnet, hinge and horseshoe. So the panel could pick a type that fails and could not reach a third of the ones that work. (value, label) PAIRS, GROUPED BY WHAT THEY ASSUME. Bare names in one alphabetical list hid the four families a user was looking for; the label says whether the fit is parametric, robust/semiparametric or rank-based and what it assumes. The stored values are unchanged, so every settings file already written goes on meaning what it meant.

### lines 285-288

```python
'regression_backend': ('combo', _regression_backend_choices(),
```

WHO fits it (instruction 141 A). Default 'statsmodels (CPU)' every existing result was produced with it, and a default that changes the numbers under a user who changed nothing is not a default. The label is the value; see _regression_backend_choices.

### lines 298-299  _(unsure)_

```python
'class_balance': ('combo', ['none', 'weighted_sampler', 'sqrt_weighted_sampler', 'weighted_loss']...
```

io.CLASS_BALANCE_MODES / io.CV_GROUP_LEVELS — both raise ValueError on anything outside these lists, so free text is not usable here.

### line 302  _(unsure)_

```python
'seg_qc': ('combo', ['off', 'report', 'flag', 'stop'], 'report'),
```

spacr.seg_qc.MODES

### lines 304-305  _(unsure)_

```python
'strict_errors': ('combo', [None, True, False], None),
```

Three states, not two: None defers to SPACR_STRICT_ERRORS so a cluster can turn it on for a batch without editing every file.

### lines 312-315

```python
'intercept': ('combo', ['fitted', 'zero', 'control', 'value'],
```

The four intercept modes, from spacr.ml.INTERCEPT_MODES. A combo rather than free text: each name selects a different construction of the design matrix, and an unrecognised one is refused at the door by prepare_formula rather than quietly fitted.

### lines 318-322

```python
'number_of_organelles': ('combo',
```

HOW MANY ORGANELLE SLOTS, as a closed list rather than a free number. The bound is real -- a slot's name is the prefix of its keys and the prefixes are lettered, so the alphabet runs out at twenty-six -- and a typed thirty would have to be clamped to a number the user did not ask for.

### lines 326-329

```python
'organelle_type': ('combo', list(_ORGANELLE_TYPE_ORDER),
```

The ONE visible organelle choice (instruction 72). A combo, not a free-text field: the nine names are a closed set, and `organelle_types.resolve_type` raises on anything else -- typing it by hand would turn a typo into a failed run instead of a pick.

### lines 343-348

```python
primary_widget_keys = tuple(
```

All slot-specific controls use the primary organelle widget contract. Generated for every slot `number_of_organelles` can name, not for the slots this run has: a settings file written at seven slots is opened by a session set to two, and its seventh slot's method must still arrive as the closed dropdown it is rather than as a free-text field whose every value fails validation.
