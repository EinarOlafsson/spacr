Workflow inputs and outputs
---------------------------

Classify
~~~~~~~~

Choose Computer Vision for crops or streamed object images, or Tabular Machine Learning for measured feature columns. The two families consume different inputs.

**Open:** Home → Classify.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Outputs**

* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Trained image classifier** — Saved classifier checkpoint, model settings and matching channel/preprocessing configuration.
* **Fitted feature classifier** — The fitted tabular classifier and its recorded feature list, training settings and validation results.
* **Classifier evaluation bundle** — Held-out predictions, labels, split metadata and calibration/leakage metrics for a saved classifier run.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Choose the image or tabular family to match your input.
* :ref:`Annotate <workflow-module-annotate>`: Keep labelled training and evaluation groups separate.
* :ref:`Gate Editor <workflow-module-gate_editor>`: Write reviewed gate selections to an annotation column, then select that same column and object population in Classify. Keep validation objects separate from training labels.
* :ref:`Image UMAP <workflow-module-umap>`: Write reviewed lasso selections to an annotation column in the matching object database, then select that column in Classify. Inspect crops and validate labels; embedding clusters are not ground truth.

**After this module**

* :ref:`Regression <workflow-module-regression>`: Select the intended CV or ML score column and preserve plate/well identity.
* :ref:`Classifier Evaluation <workflow-module-classifier_evaluation>`: Use the matching held-out predictions and split metadata.
* :ref:`Activation <workflow-module-activation>`: Also provide the matching crops and channel preprocessing.
* :ref:`Explain CV Model <workflow-module-explain_cv>`: Select CV predictions and the matching measured objects.

:doc:`API reference </api/spacr/classify/index>`.

`Module tutorial <../../../tutorials/#lesson=41_classify>`__.

