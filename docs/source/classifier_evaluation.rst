Classifier evaluation workbench
===============================

Classifier cross-validation retains every out-of-fold probability instead
of reducing a fold to one accuracy number. The workbench has no tile of its
own: it opens from the **Classify** masthead, because judging a classifier is
the sentence after training one. Press **Classifier Evaluation** there and
drop a model/results folder onto the page. The scan runs in the background and
finds every ``evaluation_manifest.json`` below that folder.

The workbench shows:

* overall accuracy, balanced accuracy, macro precision/recall/F1, log loss,
  and expected calibration error;
* raw counts and row-normalized confusion matrices for any number of classes;
* the same metrics separately for every plate;
* cross-fitted calibration curves and per-class reliability bins;
* searchable held-out predictions with plate, well, field, object, confidence,
  and raw/calibrated class probabilities; and
* an explicit leakage report for every outer and inner split.

Crop decoding and existing models
---------------------------------

Image classifiers must see the same channel order and intensity conversion
during training and prediction. New training runs use
``declared_uint8_v1`` and save it in the checkpoint's
``preprocessing.crop_loading_policy`` record. Folder and tar inputs use the
same decoder: crop formats 1 and 3 already have declared channel order;
format 2 has its channels reversed. EXIF orientation is applied before
decoding.

For unsigned 16-bit crops, decoding keeps the high byte instead of clipping
every value above 255. For example, intensities
``[0, 256, 1024, 32768, 65535]`` become ``[0, 1, 4, 128, 255]``.
This converts the classifier input to eight bits; it does not preserve the
full precision of the original scientific image or modify that source file.

Existing checkpoints without a decoding record keep ``stored_pil_v1``, the
historical PIL RGB conversion and stored channel order. spaCR reports this
fallback when loading the model. This policy can clip high-bit-depth crops,
but changing it for an already trained model would change its inputs.
Retrain and evaluate a new model to adopt the new decoding policy.
Resuming or fine-tuning retains the checkpoint's policy. Training and
validation policies must agree, and fusion or teacher models with conflicting
policies cannot be combined.

When constructing loaders through the API, pass the checkpoint policy to
``crop_loading_policy`` in :func:`spacr.io.generate_loaders` or
:func:`spacr.io.generate_cv_loaders`. Use
:func:`spacr.classification_pixels.checkpoint_policy` to read that contract;
:func:`spacr.classification_pixels.read_classification_image` and
:func:`spacr.crops.decode_crop_image` document the decoding operations.

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

Orientation stability during inference
------------------------------------------

In **Classify**, with ``classifier_family`` set to ``cv``, the **Test-time
augmentation** settings apply selected rotations and reflections when scoring
phenotype crops. This is an
inference option; it does not change training augmentation or the held-out
evaluation procedure described above. Leave it disabled when orientation is
biologically meaningful.

Enable ``tta_enabled`` and choose the transformations to evaluate:

* ``tta_rotations`` adds 90, 180 and 270 degree rotations without pixel
  interpolation.
* ``tta_horizontal_flip`` adds horizontal reflections of the selected
  rotations; ``tta_vertical_flip`` adds vertical reflections.
* The original orientation is always included. Equivalent orientations are
  evaluated once, giving at most eight views when all options are enabled.
  Enabling the feature without selecting transformations evaluates only the
  original view.

``tta_aggregation=probability_mean`` selects a label from the mean class
probabilities. ``majority_vote`` instead counts the labels of individual
views; ties prefer the higher mean probability, then the lower class index.
For binary predictions, ``score_threshold`` labels each view before agreement
and voting are calculated.

The enabled output retains both the original prediction and the aggregated
result:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Meaning
   * - ``pred``, ``prediction_mean``
     - Mean positive-class probability for binary output, or mean probability
       of the selected class for multiclass output.
   * - ``predicted_label``
     - Class selected by the configured aggregation method. The tar workflow's
       ``cv_predictions`` uses this label too.
   * - ``original_pred``, ``original_predicted_label``
     - Score and label from the original orientation before aggregation.
   * - ``prediction_std``
     - Population standard deviation across views of the positive-class
       probability for binary output, or the selected-class probability for
       multiclass output.
   * - ``transform_agreement``
     - Fraction of view labels matching the selected label.
   * - ``review_flag``, ``tta_views``
     - Whether the stability thresholds request review, and the number of
       distinct views scored.
   * - ``prob_class_<i>``, ``original_prob_class_<i>``, ``prob_class_<i>_std``
     - Mean probability, original probability and population standard
       deviation for each class index.

With majority voting, retain ``predicted_label`` or ``cv_predictions`` as
the selected result. Thresholding ``pred`` again can produce a different
label because ``pred`` still stores a mean probability.

An object is flagged when agreement is below ``tta_min_agreement`` (default
0.75) or probability standard deviation exceeds ``tta_max_std`` (default
0.15). Agreement measures stability across orientations, not calibrated
confidence or accuracy on independent data. Compare results on a separate
validation set before choosing these options for an experiment.

For directory input, pass the options as keyword arguments to
:func:`spacr.deep_spacr.apply_model`. For tar input, supply the same keys in
the settings passed to :func:`spacr.deep_spacr.apply_model_to_tar`.
:func:`spacr.inference_augmentation.transforms_for` defines the distinct
views and :func:`spacr.inference_augmentation.predict_augmented` documents
the aggregation contract. Disabled augmentation preserves ordinary inference.
