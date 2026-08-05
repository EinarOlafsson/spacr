Classifier evaluation workbench
===============================

Classifier cross-validation now retains every out-of-fold probability instead
of reducing a fold to one accuracy number. Open **Classifier Evaluation** in
the **Results & QC** section and drop a model/results folder onto it. The scan
runs in the background and finds every ``evaluation_manifest.json`` below that
folder.

The workbench shows:

* overall accuracy, balanced accuracy, macro precision/recall/F1, log loss,
  and expected calibration error;
* raw counts and row-normalized confusion matrices for any number of classes;
* the same metrics separately for every plate;
* cross-fitted calibration curves and per-class reliability bins;
* searchable held-out predictions with plate, well, field, object, confidence,
  and raw/calibrated class probabilities; and
* an explicit leakage report for every outer and inner split.

Grouped and nested cross-validation
-----------------------------------

``cross_validation_folds`` controls the outer folds and ``cv_group_by`` keeps
related fields, wells, or plates together. The default is well-grouped CV.
Set ``nested_cv_inner_folds`` to two or more to enable true nested CV.

In ordinary CV, the outer validation fold is used for checkpoint selection and
reported performance. This is fast and useful for routine comparisons, but
can give a slightly optimistic estimate when many choices are made against
that fold.

In nested CV, each outer training partition is split again. Models select
checkpoints only against an inner validation fold; the untouched outer fold is
used once for final scoring. The inner models form an ensemble for that outer
fold. This costs
``cross_validation_folds * nested_cv_inner_folds`` training runs, but keeps
model selection separate from performance estimation.

Leakage protection
------------------

Before training, spaCR checks exact paths, augmentation families, objects and
the configured grouping level on both sides of every split. With
``evaluation_fail_on_leakage=True`` (the default), any protected overlap raises
an actionable error before model fitting. Augmentations are generated after
splitting so transformed copies cannot enter a held-out fold.

Temperature calibration
-----------------------

``evaluation_calibration=temperature`` fits a scalar temperature without using
a prediction to calibrate itself. For each held-out outer fold, the temperature
is fit only from the other folds. Set it to ``none`` to retain raw model
probabilities. ``evaluation_bins`` controls the reliability table and expected
calibration error resolution.

Evaluation bundle
-----------------

The ``evaluation`` folder contains:

``oof_predictions.csv``
   One held-out row per crop, including identities and class probabilities.

``confusion_counts.csv`` and ``confusion_normalized.csv``
   Arbitrary-class confusion matrices.

``per_plate_metrics.csv`` and ``calibration.csv``
   Plate-specific quality and reliability-bin statistics.

``leakage.json``
   The auditable split checks, overlap counts, examples and warnings.

``summary.json`` and ``evaluation_manifest.json``
   Machine-readable overall results and the stable bundle schema.

The Python entry points are
:func:`spacr.classifier_evaluation.evaluate_predictions`,
:func:`spacr.classifier_evaluation.audit_split_leakage`,
:func:`spacr.classifier_evaluation.nested_group_folds`,
:func:`spacr.classifier_evaluation.write_evaluation_bundle`, and
:func:`spacr.classifier_evaluation.load_evaluation_bundle`.
