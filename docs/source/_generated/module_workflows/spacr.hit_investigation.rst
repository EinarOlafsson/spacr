Workflow inputs and outputs
---------------------------

Investigate Hit
~~~~~~~~~~~~~~~

Link a regression hit to candidate objects and quantitative well-level evidence. This does not independently validate the hit.

**Open:** Regression → Investigate Hit.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.
* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Regression <workflow-module-regression>`: Join compatible phenotype and object data for the chosen hit.

:doc:`API reference </api/spacr/hit_investigation/index>`.

