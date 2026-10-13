Workflow inputs and outputs
---------------------------

QC
~~

Inspect stored segmentation, unit, leakage and plate-effect checks. Open its nested viewers for detailed inspection.

**Open:** Home → QC.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Review available checks and missing evidence.

:doc:`API reference </api/spacr/qt/screens/qc_dashboard/index>`.

`Module tutorial <../../../../../tutorials/#lesson=54_qc_dashboard>`__.

