Workflow inputs and outputs
---------------------------

Outliers
~~~~~~~~

Review robust object/well outlier scores and the selected exclusion rule before carrying filtered data into analysis.

**Open:** QC → Outliers.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/outliers/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=66_outliers>`__.

