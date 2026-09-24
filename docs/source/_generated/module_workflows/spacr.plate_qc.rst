Workflow inputs and outputs
---------------------------

Plate Viewer
~~~~~~~~~~~~

Aggregate the selected measurement by well and inspect plate patterns and controls.

**Open:** Graph Builder → Plate Viewer.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/plate_qc/index>`.

