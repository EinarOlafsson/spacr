# Item 963: classifier choice and per-class confidence thresholds in Annotate

Requested by the maintainer on 2026-11-16:

- (a) "in the annotation app in spacr there should be an option to have spacr
  pick the best classifier. The user presses a button, e.g. a dropdown from the
  classify button, that says auto classifier, xgboost (default), resnet50."
- (b) "when classification is done I'd like an interactive per class
  thresholding mode where the user is prompted with images from the classes and
  has a slider to define the confidence threshold for each class."

Status: alpha (`ALPHA_FEATURES[963]`), hidden until Preferences > Show alpha
features is on.

## Question

1. Can Annotate pick the classifier for the user's own labels by a fair,
   pre-stated comparison (grouped held-out cross-validation, a control row), and
   record which one it picked?
2. Can the user set a confidence threshold per class while looking at that
   class's crops, and does a settings-file rerun give identical output with
   those thresholds?

## What exists today (checked in code, nightly 0d1578d9ea)

- Annotate has no button called Classify. Its model buttons are Retrain
  (`retrain_round`, default `model_type='logistic_regression'`), Suggest
  (`model_type='gradient_boosting'`, scikit-learn's
  HistGradientBoostingClassifier) and Train..., whose menu hands the source to
  the Classify (CV) or ML Analyze screens.
- `spacr.active_learning._build_round_model` builds logistic regression,
  random forest, gradient boosting and XGBoost.
- The Suggest model window already has a threshold slider (one for two classes,
  a band, minimum probabilities per class) that decides which crops get a
  *suggestion*. It does not classify every crop, and it shows no crops.

## Design

### A Classify button with a classifier menu (alpha)

A new button, Classify, in the Model group of the toolbar
(`AnnotateClassify`). Its menu:

| Entry | What it does |
|---|---|
| Auto classifier | Compares the candidate classifiers (below) on the user's labels, picks the best, classifies every crop with it and opens the thresholds window with the comparison table. |
| XGBoost (default) | Fits XGBoost on every label and classifies every crop. |
| ResNet50 on the images… | Hands the source to the Classify (CV) screen with `model_type='resnet50'`, as Train... > On the images does today. |

XGBoost is the default of this new button only. Retrain and Suggest keep their
current models, so no existing run changes its output (see Decisions).

### Auto: what is compared and how the winner is chosen

- Candidates (tabular, on the measured features Suggest uses): XGBoost,
  gradient boosting (HistGradientBoosting), LightGBM, random forest, extra
  trees and logistic regression. ResNet50 is a candidate only when a CUDA
  device is visible and has at least 4 GB free; otherwise it is skipped and
  the table says so (no free GPU). The ResNet50 candidate is the ImageNet
  ResNet50 used as a fixed feature extractor with a logistic-regression head,
  scored on the same folds; fine-tuning stays in the Classify (CV) screen.
- Folds: 5-fold grouped, stratified cross-validation
  (`spacr.io.make_cv_folds`) with the well as the group, so no well is in both
  the training and the held-out part of a fold. Fewer folds when there are
  fewer wells or a class has fewer labels. When the crops carry no well
  identity the folds are stratified but not grouped, and the table says
  "NOT grouped".
- Metric, stated before any run: macro-F1 of the pooled out-of-fold
  predictions; tie-break balanced accuracy; a remaining tie keeps the earlier
  candidate in the order above, XGBoost first.
- Control rows, never eligible to win: the majority-class baseline (every
  crop called the largest class) and a shuffled-label control (XGBoost fitted
  on labels permuted within the training part of each fold, same folds). A
  candidate that does not beat both controls is flagged.
- Recorded: the comparison table (`classifier_comparison_<column>.csv`), and
  `classifier_choice='auto'` with the resolved `classifier` in the run's
  settings file.

### Classification output

Every crop with features gets one row in
`<db folder>/annotate_classify/classification_<column>.csv`: the crop key,
one probability column per class, the most likely class, its probability, and
`class_thresholded`, which is that class when its probability reaches the
class's threshold and empty (unassigned) otherwise. The settings file
`classification_settings_<column>.csv` (and its `.json` twin) holds the
database, table, column, image filter, feature selection, seed, classifier and
`class_thresholds`. Classifying writes nothing into `measurements.db`. Save
(below) writes the thresholded class of every classified crop into `png_list`
as `<annotation column>_thresholded` (INTEGER, NULL = unassigned) and sets
`write_png_list` in the settings file, so a rerun writes the same column. The
annotation column and the Retrain/Suggest score columns are never touched.

### Per-class thresholding window

Opens when a classification finishes (`AnnotateClassThresholds`). Per class:

- a class chooser, and a slider (0.00 to 1.00, steps of 0.01) for that class's
  threshold;
- a grid of the crops the model calls that class, sorted by confidence (most
  confident first; at most 48, spread evenly over the sorted list when there
  are more), each captioned with its probability;
- as the slider moves, crops below the threshold are dimmed and captioned
  "unassigned", and the counts line updates (assigned / unassigned for the
  class and overall). No refit.
- Save writes the thresholds into the settings file and rewrites the
  classification table from the stored probabilities.

Rerunning the settings file (`spacr.active_learning._rerun_classification`)
refits the same classifier with the same seed on the same labels and applies
the same thresholds.

## Pre-set proof criteria

1. Auto picks the clear winner. Synthetic annotated set: 12 wells, 40 crops
   per well (30 labelled), two classes, 10 features with a per-well offset.
   The classes differ only along f0 - f1: f0 and f1 share a large common term
   (SD 5) and differ by +/-0.5, so either feature alone says little (a
   diagonal boundary). Logistic regression is the expected winner; it must be
   picked, with a macro-F1 at least 0.05 above every other candidate and at
   least 0.30 above both control rows. The control rows must sit near chance
   (macro-F1 <= 0.60). A second set with an XOR boundary (class = sign of
   f0 x f1) reverses the roles: Auto must not pick logistic regression there
   (its macro-F1 < 0.65) and the pick must reach macro-F1 >= 0.80.
2. No leakage: in every fold no well is in both the training and the
   held-out part (asserted on the folds used).
3. The ResNet50 candidate is skipped with a stated reason when no GPU is free
   (here: `CUDA_VISIBLE_DEVICES=''`).
4. Thresholds round-trip: classify with thresholds set, write the settings
   file, rerun it from the file: the classification table is byte-identical.
   Changing one threshold changes only the rows of that class.
5. Qt (offscreen), with real mouse events: the Classify menu offers exactly
   Auto classifier, XGBoost (default) and ResNet50 on the images…; dragging the slider with
   the mouse moves the threshold, the grid marks the crops below it as
   unassigned and the counts update; switching the class shows that class's
   crops and keeps each class's own threshold; Save writes the thresholds to
   the settings file.
6. Hidden while Show alpha features is off.

## Decisions for the maintainer

- XGBoost as the default (maintainer, 2026-11-16: Suggest and Retrain keep
  their models; XGBoost is the default of the new button only). Today Suggest fits gradient boosting
  (HistGradientBoosting) and Retrain fits logistic regression. Switching
  either to XGBoost changes their outputs, so neither was changed. XGBoost is
  the default of the new Classify button only. Should Suggest (and Retrain)
  move to XGBoost too, as a recorded default change?
- Save also writes the thresholded classes into `png_list` (maintainer,
  2026-11-16: yes), as `<annotation column>_thresholded`.
- ResNet50 in Auto (maintainer, 2026-11-16: accepted) is a fixed-feature candidate (ImageNet features + linear
  head), cheap enough to cross-validate; fine-tuned ResNet50 stays in the
  Classify (CV) screen.
