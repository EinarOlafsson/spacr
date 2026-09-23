Workflow inputs and outputs
---------------------------

Control Charts
~~~~~~~~~~~~~~

Compare designated controls across plates and inspect drift; the definition of a control comes from the experiment.

**Open:** QC → Control Charts.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/control_chart/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=51_control_charts>`__.

