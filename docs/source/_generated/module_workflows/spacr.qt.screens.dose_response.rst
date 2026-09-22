Workflow inputs and outputs
---------------------------

Dose–Response
~~~~~~~~~~~~~

Supply dose and response columns, controls and units before curve fitting; this route does not require sequencing.

**Open:** Home → Dose–Response.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Experiment Design <workflow-module-experiment_design>`: Acquire and measure the experiment first, preserving dose/condition identities.
* :ref:`Measure <workflow-module-measure>`: Join the measured response to explicit doses and controls.

:doc:`API reference </api/spacr/qt/screens/dose_response/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=69_dose_response>`__.

