Workflow inputs and outputs
---------------------------

Classifier Evaluation
~~~~~~~~~~~~~~~~~~~~~

Inspect held-out predictions, calibration and split/leakage evidence for the matching classifier.

**Open:** Classify → Classifier Evaluation.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Classifier evaluation bundle** — Held-out predictions, labels, split metadata and calibration/leakage metrics for a saved classifier run.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Classify <workflow-module-classify_merged>`: Use the matching held-out predictions and split metadata.

:doc:`API reference </api/spacr/classifier_evaluation/index>`.

`Module tutorial <../../../tutorials/#lesson=39_classifier_evaluation>`__.

