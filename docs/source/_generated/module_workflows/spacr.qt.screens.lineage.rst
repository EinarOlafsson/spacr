Workflow inputs and outputs
---------------------------

Lineage
~~~~~~~

Inspect recorded cell/nucleus/pathogen/organelle containment links; this is object containment, not time-series lineage inference.

**Open:** Help search → Database Browser → Lineage.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Linked object selection** — In memory only: selected object identities shared with compatible open views; matching crops can be opened for inspection. No figures, tables or other files are written.

:doc:`API reference </api/spacr/qt/screens/lineage/index>`.

`Module tutorial <../../../../../tutorials/#lesson=56_lineage>`__.

