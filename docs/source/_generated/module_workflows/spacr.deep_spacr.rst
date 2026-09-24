Workflow inputs and outputs
---------------------------

Activation
~~~~~~~~~~

Apply the matching image classifier and preprocessing to inspect saliency/model-response maps.

**Open:** Classify → Activation.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Trained image classifier** — Saved classifier checkpoint, model settings and matching channel/preprocessing configuration.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Image attribution maps** — Saved model-response/saliency images for the matching classifier and input preprocessing.

**Before this module**

* :ref:`Classify <workflow-module-classify_merged>`: Also provide the matching crops and channel preprocessing.

:doc:`API reference </api/spacr/deep_spacr/index>`.

`Module tutorial <../../../tutorials/#lesson=16_activation>`__.

Computer Vision
~~~~~~~~~~~~~~~

Train an image classifier from labelled object crops, archives or supported streamed crops. Match channels and preprocessing when applying its checkpoint.

**Open:** Classify → Computer Vision.

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
* **Classifier evaluation bundle** — Held-out predictions, labels, split metadata and calibration/leakage metrics for a saved classifier run.

:doc:`API reference </api/spacr/deep_spacr/index>`.

`Module tutorial <../../../tutorials/#lesson=10_classify_cv>`__.

