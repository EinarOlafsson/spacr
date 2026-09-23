Workflow inputs and outputs
---------------------------

Explain CV Model
~~~~~~~~~~~~~~~~

Fit a measured-feature surrogate to existing computer-vision scores, preserving identity joins. A surrogate explanation is not causality.

**Open:** Classify → Explain CV Model.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.

**Outputs**

* **Fitted feature classifier** — The fitted tabular classifier and its recorded feature list, training settings and validation results.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Classify <workflow-module-classify_merged>`: Select CV predictions and the matching measured objects.

:doc:`API reference </api/spacr/surrogate/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=70_explain_cv>`__.

